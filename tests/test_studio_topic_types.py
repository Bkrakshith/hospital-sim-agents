"""The Robot page's topic table says what each topic carries, as docs/robot-topics.md does (re-QA bug 19)."""
from wardzero_agents.robots.profile import get_profile
from wardzero_agents.studio.server import topic_table


def test_speech_say_is_plain_text_and_toward_is_a_track_id():
    t = {row["name"]: row["type"] for row in topic_table(get_profile("g1_claude"))}
    say = next(n for n in t if n.endswith("/speech/say"))
    toward = next(n for n in t if n.endswith("/speech/toward"))
    assert t[say] == "std_msgs/String (plain text)"
    assert t[toward] == "std_msgs/String (track id)"
