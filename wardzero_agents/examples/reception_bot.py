"""PLACEHOLDER ROBOT BEHAVIOUR, for exercising the system only. This is not the G1's real stack.

reception_bot stands in for the robot under test so a run has something to react to. It plays by
the robot fairness rule: it subscribes only to /g1/audio/heard (what its microphone hears) and
publishes only /g1/speech/say (plain text). It never reads the people's topics.

Its rules:
  * speech with "hello" or "excuse me" gets a greeting;
  * a question ("where", "which way", "looking for"...) naming a clinic, ward, toilet, exit or
    way out, or reception, gets directions built from the landmark labels in waiting_area.yaml
    and the sign texts in signs.yaml (never the internal ids);
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

# (words that name the place, landmark id, sign id). The ids only key into the config; they are
# never spoken. Every landmark people can be sent to has a row; the rest are internal points.
_PLACES = (
    (re.compile(r"\bclinics?\b|\boutpatients?\b"), "clinic_corridor", "sign_clinics"),
    (re.compile(r"\bwards?\b|\blifts?\b|\bvisiting\b"), "ward_doors", "sign_wards"),
    (re.compile(r"\btoilets?\b|\bloos?\b|\bbathrooms?\b|\brestrooms?\b"), "toilets", "sign_toilets"),
    (re.compile(r"\bexit\b|\bway out\b|\bget out\b|\bleave\b|\boutside\b|\bentrance\b|\bdoors? out\b"),
     "entrance", "sign_exit"),
    (re.compile(r"\breception\b|\bcheck(?:ing)? in\b|\bsign(?:ing)? in\b|\bregister\b"),
     "reception_g1", "sign_reception"),
)
# a question about a place: "where", "which way", "how do I get to", "looking for", "directions"
_ASK = re.compile(r"\bwhere\b|\bwhich way\b|\bhow (?:do|can|would) (?:i|we)\b|\blooking for\b"
                  r"|\bdirections?\b|\bfind\b|\bway to\b|\bget to\b|\bgo to\b|\btake (?:me|us)\b"
                  r"|\bneed (?:the|to get)\b")
_GREET = re.compile(r"\bhello\b|\bexcuse me\b")
_ARROWS = re.compile(r"[\u2190-\u21ff]")


def _label(text: str) -> str:
    text = re.sub(r"\s*\(.*?\)\s*", " ", text).strip()
    text = text.split(", ")[0]  # "Reception desk, where the G1 stands" -> "Reception desk"
    return text[:1].lower() + text[1:]


def _sign_text(text: str) -> str:
    return re.sub(r"\s+", " ", _ARROWS.sub("", text)).strip()


class ReceptionBot:
    def __init__(self, area: WaitingArea, signs: list[Sign]):
        signs_by_id = {s.id: s for s in signs}
        self._directions: list[tuple[re.Pattern, str]] = []
        for pattern, landmark, sign_id in _PLACES:
            lm = area.landmarks.get(landmark)
            if lm is None:
                continue
            # Spoken names come from the config labels and sign texts, never from the ids.
            sign = signs_by_id.get(sign_id)
            if landmark == "reception_g1":
                line = "You're already at the reception desk. Tell me what you need and I'll do what I can."
            else:
                line = f"Head for the {_label(lm.label)}."
                if sign:
                    line += f" Follow the sign that says \"{_sign_text(sign.text)}\"."
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
        if _ASK.search(text):
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
