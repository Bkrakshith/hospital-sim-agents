"""Robot fairness audit: no node outside the wardzero_ prefix may read the people's minds.

The robot under test may hear /g1/audio/heard and speak on /g1/speech/say. Anything under
FORBIDDEN_FOR_ROBOT (the people's topics, the scenario and the raw audio and world events) is
off limits to it. The audit is pure Python; live_audit only needs a node-like object, so it
imports nothing from rclpy.
"""
from __future__ import annotations

import logging
from typing import Any

from .topics import FORBIDDEN_FOR_ROBOT

log = logging.getLogger(__name__)


def _base_name(node: str) -> str:
    return node.rstrip("/").rsplit("/", 1)[-1]


def audit(subs_by_node: dict[str, list[str]], allowed_prefix: str = "wardzero_") -> list[dict]:
    """[{node, topic}] for every subscription by a non-wardzero node to a forbidden topic."""
    out: list[dict] = []
    for node, topics in subs_by_node.items():
        if _base_name(node).startswith(allowed_prefix):
            continue
        for topic in topics:
            if any(topic.startswith(p) for p in FORBIDDEN_FOR_ROBOT):
                out.append({"node": node, "topic": topic})
    return out


def _qualified(name: str, ns: str) -> str:
    if not ns or ns == "/":
        return name
    return f"{ns.rstrip('/')}/{name}"


def live_audit(node: Any, allowed_prefix: str = "wardzero_") -> list[dict]:
    """Audit the live ROS graph as seen from node (an rclpy Node)."""
    subs: dict[str, list[str]] = {}
    for name, ns in node.get_node_names_and_namespaces():
        try:
            pairs = node.get_subscriber_names_and_types_by_node(name, ns)
        except Exception as exc:  # the node left between the two calls
            log.debug("fairness audit: skipping %s (%s)", name, type(exc).__name__)
            continue
        subs[_qualified(name, ns)] = [topic for topic, _types in pairs]
    return audit(subs, allowed_prefix)
