"""PLACEHOLDER ROBOT BEHAVIOUR, for exercising the system only. This is not the G1's real stack.

reception_bot stands in for the robot under test so a run has something to react to. It plays by
the robot fairness rule: it subscribes only to /g1/audio/heard (what its microphone hears) and
publishes only /g1/speech/say (plain text). It never reads the people's topics.

Its rules:
  * speech with "hello" or "excuse me" gets a greeting;
  * "where" plus clinic, ward, toilet, exit or "way out" gets directions built from the landmark
    labels in waiting_area.yaml and the sign texts in signs.yaml;
  * three coughs received at 50 dB or more within 20 s get "Can I get some help for you? I'll let
    the nurse know.", at most once every 60 s.

Its node name is reception_bot, outside the wardzero_ prefix on purpose: the fairness audit
treats it as the robot.

    python -m wardzero_agents.examples.reception_bot --scenario FILE
"""
from __future__ import annotations

import logging
import re
import sys
from collections import deque
from typing import Optional

from wardzero_agents.config.models import Sign, WaitingArea
from wardzero_agents.contracts import Heard

log = logging.getLogger("reception_bot")

GREETING = "Hello, I'm the reception robot. How can I help you?"
HELP_LINE = "Can I get some help for you? I'll let the nurse know."
COUGH_MIN_DB = 50.0
COUGH_WINDOW_S = 20.0
COUGH_COUNT = 3
HELP_COOLDOWN_S = 60.0

# (words that ask for the place, landmark id, sign id)
_PLACES = (
    (re.compile(r"\bclinics?\b"), "clinic_corridor", "sign_clinics"),
    (re.compile(r"\bwards?\b"), "ward_doors", "sign_wards"),
    (re.compile(r"\btoilets?\b"), "toilets", "sign_toilets"),
    (re.compile(r"\bexit\b|\bway out\b"), "entrance", "sign_exit"),
)
_WHERE = re.compile(r"\bwhere\b|\bwhere's\b")
_GREET = re.compile(r"\bhello\b|\bexcuse me\b")


def _label(text: str) -> str:
    text = re.sub(r"\s*\(.*?\)\s*", " ", text).strip()
    return text[:1].lower() + text[1:]


class ReceptionBot:
    def __init__(self, area: WaitingArea, signs: list[Sign]):
        signs_by_id = {s.id: s for s in signs}
        self._directions: list[tuple[re.Pattern, str]] = []
        for pattern, landmark, sign_id in _PLACES:
            lm = area.landmarks.get(landmark)
            if lm is None:
                continue
            # The sentence names the landmark id on purpose. This phrasing is for the placeholder
            # only: it lets the offline test policy (FakeGateway) walk to the place it hears.
            line = f"That's the {_label(lm.label)}. Head for the {landmark}"
            sign = signs_by_id.get(sign_id)
            line += f" and follow the sign that says \"{sign.text}\"." if sign else "."
            self._directions.append((pattern, line))
        self._coughs: deque[float] = deque()
        self._last_help: Optional[float] = None

    def on_heard(self, h: Heard, t: float) -> Optional[str]:
        """What the robot says in reply to one thing it heard, or None."""
        if h.source_kind == "cough":
            return self._cough(h, t)
        if h.source_kind != "person" or not h.text:
            return None
        text = h.text.lower()
        if _WHERE.search(text):
            for pattern, line in self._directions:
                if pattern.search(text):
                    return line
        if _GREET.search(text):
            return GREETING
        return None

    def _cough(self, h: Heard, t: float) -> Optional[str]:
        if h.received_db < COUGH_MIN_DB:
            return None
        self._coughs.append(t)
        while self._coughs and self._coughs[0] < t - COUGH_WINDOW_S:
            self._coughs.popleft()
        if len(self._coughs) < COUGH_COUNT:
            return None
        if self._last_help is not None and t - self._last_help < HELP_COOLDOWN_S:
            return None
        self._last_help = t
        self._coughs.clear()
        return HELP_LINE


# ---- ROS wrapper ---------------------------------------------------------------------------
def _parse(argv):
    import argparse
    p = argparse.ArgumentParser(description="Placeholder reception robot (hears /g1/audio/heard only).")
    p.add_argument("--scenario", required=True, help="scenario file (for the area landmarks and signs)")
    return p.parse_args(argv)


async def _run(args) -> int:
    import asyncio

    from wardzero_agents.config.loader import ConfigError, load_scenario
    from wardzero_agents.ros import topics
    from wardzero_agents.ros.rosutil import RosBridge, install_stop_signals

    try:
        loaded = load_scenario(args.scenario)
    except ConfigError as exc:
        print(f"Scenario error: {exc}", file=sys.stderr)
        return 2
    bot = ReceptionBot(loaded.area, loaded.signs)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    install_stop_signals(loop, stop)
    ros = RosBridge("reception_bot", loop)

    def on_heard(h: Heard) -> None:
        reply = bot.on_heard(h, ros.now())
        if reply:
            log.info("says: %s", reply)
            ros.publish_text(topics.G1_SAY, reply)

    ros.subscribe_json(Heard, topics.G1_HEARD, on_heard)
    ros.publisher(topics.G1_SAY)
    ros.start()
    log.info("reception_bot up (placeholder robot)")
    try:
        await stop.wait()
    finally:
        ros.close()
    return 0


def main(argv=None) -> int:
    import asyncio

    from wardzero_agents.ros.rosutil import setup_logging
    setup_logging()
    return asyncio.run(_run(_parse(argv)))


if __name__ == "__main__":
    sys.exit(main())
