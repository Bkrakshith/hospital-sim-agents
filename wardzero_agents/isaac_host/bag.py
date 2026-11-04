"""A rosbag of the robot's ROS domain, recorded next to the run (``<run>/robot_bag``).

``ros2 bag record`` of an explicit list of the robot's topics runs as a child of the Isaac host, in the system's Humble (Python 3.10), not Isaac's bundled
rclpy (Python 3.12): the host's own environment is stripped and the system's is sourced from ``env.sh``. The list is the robot's own
topics (no images unless asked: rgb and depth were 90% of a raw bag); messages are zstd-compressed. The bag is
finalised by SIGINT (its ``metadata.yaml`` is written then), so ``stop`` always signals the child's process group and
waits; the child also asks the kernel for a SIGINT of its own when the host dies, so it is never left orphaned.
"""
from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("wardzero.isaac.bag")

BAG_DIR = "robot_bag"
STOP_WAIT_S = 20.0  # after SIGINT, before SIGTERM
TERM_WAIT_S = 5.0  # after SIGTERM, before SIGKILL
from ..isaac_assets import ros_env_sh

DEFAULT_ENV_SH = ros_env_sh()
# nothing of the host's (Isaac libraries, conda, other ROS settings, keys) may reach the system's ROS
DROP_ENV_PREFIXES = ("CONDA", "ROS_", "RMW_", "AMENT_", "COLCON_")
DROP_ENV = ("LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
            "ANTHROPIC_AUTH_TOKEN", "WARDZERO_KEY_FILE")


def bag_env(domain: int, base: Optional[dict] = None) -> dict:
    env = {k: v for k, v in (os.environ if base is None else base).items()
           if k not in DROP_ENV and not k.startswith(DROP_ENV_PREFIXES)}
    env["ROS_DOMAIN_ID"] = str(int(domain))
    return env


IMAGE_KEYS = ("camera_rgb", "camera_depth", "lidar")  # big: only with --bag-images
BAG_KEYS = ("clock", "tf", "joint_states", "joint_command", "odom", "cmd_vel", "camera_info", "bbox_3d", "heard", "say",
            "goal_pose", "task_action", "task_result", "task_brief")


def bag_topics(ns: str = "robot", images: bool = False) -> list[str]:
    """The robot's own topics, from the one table (``ros.topics.robot_topics``): never a wildcard."""
    from ..ros import topics

    rt = topics.robot_topics(ns)
    return [rt[k] for k in BAG_KEYS + (IMAGE_KEYS if images else ())]


def bag_argv(env_sh: Path, out_dir: Path, topic_list: list[str]) -> list[str]:
    """``setpriv`` asks the kernel for a SIGINT when the host dies; bash sources the system ROS, then becomes
    ``ros2 bag record`` (so the signal reaches it)."""
    return ["setpriv", "--pdeathsig", "INT", "--", "bash", "-c",
            'source "$1" >/dev/null 2>&1 && o="$2" && shift 2 && exec ros2 bag record -o "$o" '
            '--compression-mode message --compression-format zstd "$@"',
            "wardzero-bag", str(env_sh), str(out_dir), *topic_list]


def _killpg(proc, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)  # the child leads its process group
    except (ProcessLookupError, PermissionError):
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass


class BagRecorder:
    def __init__(self, run_dir: Path, domain: int, private_domain: int, ns: str = "robot", images: bool = False,
                 env_sh: Optional[Path] = None, popen: Callable = subprocess.Popen,
                 signaller: Callable = _killpg, stop_wait_s: float = STOP_WAIT_S, term_wait_s: float = TERM_WAIT_S):
        self.out_dir = Path(run_dir) / BAG_DIR
        self.domain, self.private_domain = int(domain), int(private_domain)
        self.topics = bag_topics(ns, images)
        self.env_sh = Path(env_sh or os.environ.get("WARDZERO_ROS_ENV_SH") or DEFAULT_ENV_SH)
        self._popen, self._signal = popen, signaller
        self._stop_wait, self._term_wait = stop_wait_s, term_wait_s
        self.proc = None
        self.result: Optional[str] = None  # "clean", "terminated" or "killed" after stop()

    def start(self) -> bool:
        """Start recording; False (and a warning) when it cannot start: a run is never failed by its bag."""
        if self.proc is not None:
            return True
        if self.domain == self.private_domain:
            log.error("no rosbag: the robot is on the private domain (%d), where the people's minds and the director "
                      "live; give the robot a domain of its own (--robot-domain)", self.domain)
            return False
        if not self.env_sh.is_file():
            log.warning("WARNING: no rosbag will be recorded: %s is missing. Source of the system's ROS 2 Humble; "
                        "set WARDZERO_ROS_ENV_SH to your own env file.", self.env_sh)
            return False
        try:
            self.proc = self._popen(bag_argv(self.env_sh, self.out_dir, self.topics), env=bag_env(self.domain),
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    start_new_session=True)
        except OSError as exc:
            log.warning("no rosbag: %s", exc)
            return False
        atexit.register(self.stop)
        log.info("rosbag of domain %d into %s", self.domain, self.out_dir)
        return True

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def finalised(self) -> bool:
        return (self.out_dir / "metadata.yaml").is_file()

    def stop(self) -> Optional[str]:
        """SIGINT, wait for the bag to be written; SIGTERM and SIGKILL only if it hangs. Safe to call twice."""
        proc, self.proc = self.proc, None
        if proc is None:
            return self.result
        try:
            atexit.unregister(self.stop)
        except Exception:
            pass
        if proc.poll() is not None:
            self.result = "clean" if proc.returncode in (0, -int(signal.SIGINT)) else "exited early"
            return self.result
        self._signal(proc, signal.SIGINT)
        try:
            proc.wait(timeout=self._stop_wait)
            self.result = "clean"
        except subprocess.TimeoutExpired:
            self._signal(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=self._term_wait)
                self.result = "terminated"
            except subprocess.TimeoutExpired:
                self._signal(proc, signal.SIGKILL)
                proc.wait()
                self.result = "killed"
            log.warning("the rosbag did not stop on SIGINT (%s); it may be incomplete", self.result)
        return self.result
