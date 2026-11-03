"""ROS 2 topic names for the sim boundary. Pure strings and functions, no rclpy."""

ROSTER = "/humans/roster"

AUDIO_EVENTS = "/sim/audio/events"
G1_HEARD = "/g1/audio/heard"
G1_SAY = "/g1/speech/say"
WORLD_EVENT = "/sim/world_event"
TIMELINE = "/scenario/timeline"
SCEN_EVENTS = "/scenario/events"
CONTROL = "/scenario/control"
PENDING = "/agents/pending"

FORBIDDEN_FOR_ROBOT = ("/humans/", "/scenario/", "/sim/audio/events", "/sim/world_event")


def status(pid: str) -> str:
    return f"/humans/{pid}/status"


def intent(pid: str) -> str:
    return f"/humans/{pid}/intent"


def observation(pid: str) -> str:
    return f"/humans/{pid}/observation"


def say(pid: str) -> str:
    return f"/humans/{pid}/say"


def nudge(pid: str) -> str:
    return f"/humans/{pid}/nudge"


def reflex(pid: str) -> str:
    return f"/humans/{pid}/reflex"
