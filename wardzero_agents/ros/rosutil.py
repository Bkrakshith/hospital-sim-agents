"""rclpy plumbing shared by the nodes: QoS, a background executor thread and the asyncio hand-off.

Threading model, the same in every node: rclpy spins a MultiThreadedExecutor in a background
thread; the asyncio loop runs in the main thread and owns all node state. ROS callbacks only
parse the message and hand it to the loop with call_soon_threadsafe, so state is never touched
from two threads. Publishing happens on the loop thread.

The Isaac host has no asyncio loop: its bridges are built with loop=None and an explicit
domain_id (one rclpy Context per domain, S1 Q3), callbacks are queued and the frame loop runs them
with drain().

Only the nodes' main() functions import this module, so the service classes stay testable
without a sourced ROS environment.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import queue
import signal
import threading
from typing import Any, Callable, Optional

import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

from ..contracts import ContractError, from_json, to_json
from ..keys import install_log_redaction

log = logging.getLogger(__name__)

QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=50, reliability=ReliabilityPolicy.RELIABLE,
                 durability=DurabilityPolicy.VOLATILE)
QOS_LATCHED = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)


def setup_logging() -> None:
    install_log_redaction()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def base_args(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--run-dir", required=True, help="run folder to record into")
    return p


def make_node(name: str, use_sim_time: bool = True, context=None) -> Node:
    return Node(name, context=context, parameter_overrides=[
        Parameter("use_sim_time", Parameter.Type.BOOL, use_sim_time)])


def sim_now(node: Node) -> float:
    return node.get_clock().now().nanoseconds / 1e9


class RosBridge:
    """One node, its executor thread and the asyncio loop it hands callbacks to.

    With loop=None the callbacks wait in a queue until drain() runs them on the caller's thread.
    With a domain_id the bridge has its own rclpy Context on that domain (the default context and
    ROS_DOMAIN_ID otherwise)."""

    def __init__(self, name: str, loop: Optional[asyncio.AbstractEventLoop], use_sim_time: bool = True,
                 domain_id: Optional[int] = None):
        self.context = None
        if domain_id is None:
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        else:
            self.context = rclpy.Context()
            rclpy.init(context=self.context, domain_id=int(domain_id),
                       signal_handler_options=SignalHandlerOptions.NO)
        self.domain_id = domain_id
        self.node = make_node(name, use_sim_time, self.context)
        self.loop = loop
        self._queue: "queue.SimpleQueue[tuple[Callable, tuple]]" = queue.SimpleQueue()
        self._pubs: dict[str, Any] = {}
        self._subs: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._bad: dict[str, int] = {}
        self.executor = MultiThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self._thread = threading.Thread(target=self._spin, name=f"{name}-spin", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _spin(self) -> None:
        try:
            self.executor.spin()
        except ExternalShutdownException:
            pass
        except Exception as exc:  # keep the process alive; the main loop decides when to stop
            log.error("ROS executor stopped: %s", type(exc).__name__)

    def now(self) -> float:
        return sim_now(self.node)

    # ---- hand-off ------------------------------------------------------------------------
    def _to_loop(self, fn: Callable, *args) -> None:
        if self.loop is None:
            self._queue.put((fn, args))
            return
        try:
            self.loop.call_soon_threadsafe(fn, *args)
        except RuntimeError:
            pass  # the loop is closing

    def drain(self, limit: int = 10000) -> int:
        """Run the queued callbacks (loop=None bridges). Returns how many ran."""
        n = 0
        while n < limit:
            try:
                fn, args = self._queue.get_nowait()
            except queue.Empty:
                break
            n += 1
            try:
                fn(*args)
            except Exception:  # a bad message must not stop the frame loop (asyncio logs and goes on too)
                log.exception("ROS callback failed")
        return n

    def _warn_bad(self, topic: str, exc: Exception) -> None:
        n = self._bad.get(topic, 0) + 1
        self._bad[topic] = n
        if n == 1 or n % 100 == 0:
            log.warning("dropped an invalid message on %s (%d so far): %s", topic, n, str(exc)[:200])

    def subscribe_json(self, cls, topic: str, handler: Callable, qos: QoSProfile = QOS) -> None:
        """Parse wz.v1 JSON in the ROS thread, call handler(model) on the loop thread."""
        with self._lock:
            if topic in self._subs:
                return

            def cb(msg: String) -> None:
                try:
                    model = from_json(cls, msg.data)
                except ContractError as exc:
                    self._warn_bad(topic, exc)
                    return
                self._to_loop(handler, model)

            self._subs[topic] = self.node.create_subscription(String, topic, cb, qos)

    def subscribe(self, msg_type, topic: str, handler: Callable, qos: QoSProfile = QOS,
                  convert: Optional[Callable] = None) -> None:
        with self._lock:
            if topic in self._subs:
                return

            def cb(msg) -> None:
                self._to_loop(handler, convert(msg) if convert else msg)

            self._subs[topic] = self.node.create_subscription(msg_type, topic, cb, qos)

    def publisher(self, topic: str, msg_type=String, qos: QoSProfile = QOS):
        with self._lock:
            pub = self._pubs.get(topic)
            if pub is None:
                pub = self._pubs[topic] = self.node.create_publisher(msg_type, topic, qos)
            return pub

    def publish_json(self, topic: str, model, qos: QoSProfile = QOS) -> None:
        msg = String()
        msg.data = to_json(model)
        self.publisher(topic, String, qos).publish(msg)

    def publish_text(self, topic: str, text: str, qos: QoSProfile = QOS) -> None:
        msg = String()
        msg.data = text
        self.publisher(topic, String, qos).publish(msg)

    def close(self) -> None:
        try:
            self.executor.shutdown(timeout_sec=2.0)
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass
        if self.context is not None:
            if self.context.ok():
                rclpy.shutdown(context=self.context)
        elif rclpy.ok():
            rclpy.shutdown()
        if self._thread.ident is not None:  # started
            self._thread.join(timeout=2.0)


def install_stop_signals(loop: asyncio.AbstractEventLoop, stop: asyncio.Event) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)


async def wait_or_sleep(stop: asyncio.Event, seconds: float) -> None:
    """Sleep, waking early when stop is set."""
    try:
        await asyncio.wait_for(stop.wait(), seconds)
    except asyncio.TimeoutError:
        pass
