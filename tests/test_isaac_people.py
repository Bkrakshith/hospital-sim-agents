"""Intents to Behavior System calls, with a fake Behavior API (no Isaac): the mapping, the re-issue of failed walks,
the sit and stand sequence, the head turns, and posture from the behavior state."""
import math
from pathlib import Path

import pytest

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.contracts import Intent
from wardzero_agents.isaac_host import people as pe
from wardzero_agents.isaac_host.bodies import IsaacBodies
from wardzero_agents.ros.sim_service import SimService
from wardzero_agents.runlog import RunLog

V1 = Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml"
DT = 1.0 / 30.0


class FakeApi:
    """A character that walks straight at its set speed. fail_walks: how many MoveTos end short and silently."""

    def __init__(self, x, y, yaw=0.0, seats=None, fail_walks=0, freeze=False):
        self.x, self.y, self.yaw = x, y, yaw
        self.calls: list[tuple] = []
        self.tasks: dict[int, dict] = {}
        self.next_id = 1
        self.speed = 1.5
        self.action = "Idle"
        self.cur = None
        self.fail_walks = fail_walks
        self.freeze = freeze
        self.hidden = False
        self.seats = seats or {}
        self.t = 0.0

    def _new(self, kind, **kw):
        tid = self.next_id
        self.next_id += 1
        if self.cur is not None and self.tasks[self.cur]["state"] == "running":
            self.tasks[self.cur]["state"] = "failed"  # a new task cancels the old one
        self.tasks[tid] = {"kind": kind, "state": "running", "age": 0.0, **kw}
        self.cur = tid
        self.action = kind
        return tid

    def move_to(self, x, y):
        self.calls.append(("move_to", round(x, 2), round(y, 2)))
        return self._new("MoveTo", goal=(x, y), fail=self.fail_walks > 0)

    def sit(self, prop):
        self.calls.append(("sit", prop))
        return self._new("Sit", prop=prop)

    def idle(self):
        self.calls.append(("idle",))
        return self._new("Idle")

    def look_at(self, x, y, z, duration):
        self.calls.append(("look_at", round(x, 1), round(y, 1), round(z, 2)))
        return self._new("LookAt")

    def cancel(self, tid):
        self.calls.append(("cancel", tid))
        if tid in self.tasks:
            self.tasks[tid]["state"] = "failed"

    def status(self, tid):
        return self.tasks[tid]["state"] if tid in self.tasks else "none"

    def pose(self):
        return (self.x, self.y, self.yaw)

    def current_action(self):
        return self.action

    def set_speed(self, v):
        self.calls.append(("set_speed", round(v, 2)))
        self.speed = v

    def hide(self):
        self.hidden = True
        self.calls.append(("hide",))

    def tick(self, dt):
        self.t += dt
        if self.cur is None:
            return
        tk = self.tasks[self.cur]
        tk["age"] += dt
        if tk["kind"] == "MoveTo" and tk["state"] == "running" and not self.freeze:
            if tk["fail"] and tk["age"] > 0.3:  # crowd avoidance gave up: ended short, no error
                tk["state"] = "failed"
                self.fail_walks -= 1
                self.action = "Idle"
                return
            gx, gy = tk["goal"]
            d = math.hypot(gx - self.x, gy - self.y)
            step = self.speed * dt
            self.yaw = math.atan2(gy - self.y, gx - self.x)
            if d <= step:
                self.x, self.y = gx, gy
                tk["state"] = "succeeded"
                self.action = "Idle"
            else:
                self.x += (gx - self.x) / d * step
                self.y += (gy - self.y) / d * step
        elif tk["kind"] == "Sit" and tk["state"] == "running" and tk["age"] > 1.0:
            sx, sy, syaw = self.seats[tk["prop"]]
            self.x, self.y, self.yaw = sx, sy, syaw


def make(tmp_path, people=("P5",), **api_kw):
    loaded = load_scenario(V1)
    bodies = IsaacBodies()
    svc = SimService(loaded, RunLog(tmp_path), 1.0, False, (), bodies=bodies)
    seats = {pe.seat_prim_path(s.id): (s.x, s.y, s.yaw) for s in svc.world.site.seats.values()}
    apis = {}
    for pid in people:
        p = svc.world.people[pid]
        api = FakeApi(p.x, p.y, p.yaw, seats=seats, **api_kw)
        apis[pid] = api
        ctrl = pe.PersonController(pid, api)
        bodies.add_controller(svc.world, ctrl)
        ctrl.start(svc.world)
    return svc, bodies, apis


def run(svc, apis, seconds):
    for _ in range(int(seconds / DT)):
        for a in apis.values():
            a.tick(DT)
        svc.step(DT)


def status(svc, pid):
    return next(s for s in svc.world.statuses() if s.person == pid)


def intent(pid, kind, **kw):
    return Intent(id=f"{kind}-{pid}", person=pid, t=0.0, kind=kind, **kw)


def test_a_person_who_starts_seated_sits_on_their_seat(tmp_path):
    svc, _, apis = make(tmp_path, ("P5",))
    assert apis["P5"].calls == [("sit", "/World/Seats/D2")]


def test_walk_to_is_a_move_to_the_resolved_place(tmp_path):
    svc, bodies, apis = make(tmp_path)
    svc.step(DT)
    svc.on_intent(intent("P5", "walk_to", target="entrance"))
    run(svc, apis, 0.2)
    calls = apis["P5"].calls
    assert calls[1] == ("idle",)  # the sitter stands first
    run(svc, apis, 3.0)
    moves = [c for c in apis["P5"].calls if c[0] == "move_to"]
    ent = svc.world.site.resolve("entrance")
    assert moves and moves[0][1:] == pytest.approx(ent, abs=0.01)


def test_the_walk_ends_the_intent_and_the_pose_is_the_characters(tmp_path):
    svc, _, apis = make(tmp_path)
    svc.step(DT)
    svc.on_intent(intent("P5", "walk_to", target="entrance"))
    mid = None
    for k in range(int(60 / DT)):
        for a in apis.values():
            a.tick(DT)
        svc.step(DT)
        if k == int(4 / DT):
            mid = status(svc, "P5")
        if status(svc, "P5").state == "done":
            break
    st = status(svc, "P5")
    assert st.state == "done" and st.posture == "standing" and st.progress == 1.0
    api = apis["P5"]
    assert (st.x, st.y) == pytest.approx((api.x, api.y))  # the world's pose is the character's
    assert math.hypot(st.x - svc.world.site.resolve("entrance")[0], st.y - svc.world.site.resolve("entrance")[1]) < 0.5
    assert mid is not None and mid.state == "running" and 0.0 <= mid.progress < 1.0


def test_posture_follows_the_behavior_state(tmp_path):
    svc, _, apis = make(tmp_path)
    svc.step(DT)
    svc.on_intent(intent("P5", "walk_to", target="entrance"))
    seen = set()
    for _ in range(int(20 / DT)):
        apis["P5"].tick(DT)
        svc.step(DT)
        seen.add(status(svc, "P5").posture)
    assert {"seated", "walking", "standing"} <= seen


def test_a_dodge_during_a_walk_still_counts_as_walking(tmp_path):
    svc, _, apis = make(tmp_path)
    svc.step(DT)
    svc.on_intent(intent("P5", "walk_to", target="entrance"))
    run(svc, apis, 3.0)
    assert status(svc, "P5").posture == "walking"
    apis["P5"].action = "Dodge"  # crowd avoidance steps aside; the MoveTo goes on
    svc.step(DT)
    assert status(svc, "P5").posture == "walking"
    apis["P5"].action = "Idle"
    svc.step(DT)
    assert status(svc, "P5").posture == "standing"


def test_sit_down_stands_walks_to_the_floor_in_front_of_the_seat_and_sits(tmp_path):
    svc, _, apis = make(tmp_path)
    svc.step(DT)
    svc.on_intent(intent("P5", "sit_down", target="A1"))
    run(svc, apis, 40.0)
    calls = [c for c in apis["P5"].calls if c[0] in ("idle", "move_to", "sit")]
    assert [c[0] for c in calls] == ["sit", "idle", "move_to", "sit"]
    seat = svc.world.site.seats["A1"]
    front = pe.seat_approach(seat)
    assert calls[2][1:] == pytest.approx(front, abs=0.01)
    assert math.hypot(front[0] - seat.x, front[1] - seat.y) == pytest.approx(pe.SEAT_APPROACH_M)
    assert calls[3] == ("sit", "/World/Seats/A1")
    st = status(svc, "P5")
    assert st.state == "done" and st.posture == "seated"
    assert svc.world.people["P5"].seat == "A1"


def test_stand_up_issues_idle_and_waits_for_the_stand(tmp_path):
    svc, _, apis = make(tmp_path)
    svc.step(DT)
    svc.on_intent(intent("P5", "stand_up"))
    run(svc, apis, 0.5)
    assert status(svc, "P5").state == "running" and status(svc, "P5").posture == "seated"
    run(svc, apis, 6.0)
    assert status(svc, "P5").state == "done" and status(svc, "P5").posture == "standing"
    assert svc.world.people["P5"].seat is None


def test_a_walk_that_ended_short_is_reissued_until_it_arrives(tmp_path):
    svc, _, apis = make(tmp_path, fail_walks=2)
    svc.step(DT)
    svc.on_intent(intent("P5", "walk_to", target="entrance"))
    run(svc, apis, 90.0)
    moves = [c for c in apis["P5"].calls if c[0] == "move_to"]
    assert len(moves) == 3  # the first try and two re-issues
    assert status(svc, "P5").state == "done"


def test_a_walk_that_keeps_failing_ends_the_intent_blocked(tmp_path):
    svc, _, apis = make(tmp_path, fail_walks=99)
    svc.step(DT)
    svc.on_intent(intent("P5", "walk_to", target="entrance"))
    run(svc, apis, 90.0)
    moves = [c for c in apis["P5"].calls if c[0] == "move_to"]
    assert len(moves) == 1 + pe.WALK_RETRIES
    st = status(svc, "P5")
    assert st.state == "failed" and st.reason == "blocked"
    assert svc.world.people["P5"].plan == []
    assert svc.world.people["P5"].claim is None


def test_a_character_that_stops_moving_counts_as_blocked_in_the_end(tmp_path):
    svc, _, apis = make(tmp_path, freeze=True)
    svc.step(DT)
    svc.on_intent(intent("P5", "walk_to", target="entrance"))
    run(svc, apis, 120.0)
    st = status(svc, "P5")
    assert st.state == "failed" and st.reason == "blocked"


def test_a_blocked_walk_before_a_sit_frees_the_claim(tmp_path):
    svc, _, apis = make(tmp_path, fail_walks=99)
    svc.step(DT)
    svc.on_intent(intent("P5", "sit_down", target="A1"))
    run(svc, apis, 90.0)
    assert status(svc, "P5").reason == "blocked"
    svc2, _, _ = make(tmp_path / "b", ("P1",))
    svc2.world.people["P1"].seat = None
    assert not svc.world._seat_taken("A1", "P1")


def test_look_at_a_person_turns_the_head_when_standing(tmp_path):
    svc, _, apis = make(tmp_path, ("P5", "P1"))
    svc.step(DT)
    svc.on_intent(intent("P5", "stand_up"))
    run(svc, apis, 5.0)
    n = len(apis["P5"].calls)
    svc.on_intent(intent("P5", "look_at", target="P1"))
    new = apis["P5"].calls[n:]
    p1 = svc.world.people["P1"]
    assert new and new[0][0] == "look_at" and new[0][1:3] == pytest.approx((p1.x, p1.y), abs=0.1)


def test_a_seated_person_is_never_sent_a_head_turn_that_would_stand_them_up(tmp_path):
    svc, _, apis = make(tmp_path, ("P5", "P1"))
    svc.step(DT)
    n = len(apis["P5"].calls)
    svc.on_intent(intent("P5", "look_at", target="P1"))
    svc.on_intent(intent("P5", "fidget"))
    svc.on_intent(intent("P5", "read_sign", target="reception_front"))
    run(svc, apis, 5.0)
    assert apis["P5"].calls[n:] == []


def test_a_rejected_intent_is_not_forwarded(tmp_path):
    svc, bodies, apis = make(tmp_path)
    svc.step(DT)
    n = len(apis["P5"].calls)
    svc.on_intent(intent("P5", "walk_to", target="nowhere_land"))
    assert apis["P5"].calls[n:] == []
    assert status(svc, "P5").state == "failed"


def test_fidget_is_a_look_around_sweep_of_look_at_targets(tmp_path):
    svc, _, apis = make(tmp_path)
    svc.step(DT)
    svc.on_intent(intent("P5", "stand_up"))
    run(svc, apis, 5.0)
    n = len(apis["P5"].calls)
    svc.on_intent(intent("P5", "fidget"))
    run(svc, apis, 5.0)
    looks = [c for c in apis["P5"].calls[n:] if c[0] == "look_at"]
    assert len(looks) == len(pe.SWEEP_DEG)
    assert len({(c[1], c[2]) for c in looks}) == len(pe.SWEEP_DEG)


def test_a_walk_interrupts_a_sweep(tmp_path):
    svc, _, apis = make(tmp_path)
    svc.step(DT)
    svc.on_intent(intent("P5", "stand_up"))
    run(svc, apis, 5.0)
    svc.on_intent(intent("P5", "fidget"))
    svc.on_intent(intent("P5", "walk_to", target="entrance"))
    n = len(apis["P5"].calls)
    run(svc, apis, 5.0)
    assert not [c for c in apis["P5"].calls[n:] if c[0] == "look_at"]


def test_talk_turns_the_head_to_the_addressee(tmp_path):
    svc, _, apis = make(tmp_path, ("P5", "P1"))
    svc.step(DT)
    svc.on_intent(intent("P5", "stand_up"))
    run(svc, apis, 5.0)
    n = len(apis["P5"].calls)
    svc.on_intent(intent("P5", "talk", text="hello", addressee="P1"))
    assert [c[0] for c in apis["P5"].calls[n:]] == ["look_at"]


def test_leaving_hides_the_character_at_the_exit(tmp_path):
    svc, _, apis = make(tmp_path)
    svc.step(DT)
    svc.on_intent(intent("P5", "leave"))
    run(svc, apis, 120.0)
    assert svc.world.people["P5"].posture == "gone"
    assert apis["P5"].hidden


def test_approach_chases_a_person_who_moves(tmp_path):
    svc, _, apis = make(tmp_path, ("P5", "P1"))
    svc.step(DT)
    svc.on_intent(intent("P5", "stand_up"))
    svc.on_intent(intent("P1", "stand_up"))
    run(svc, apis, 5.0)
    svc.on_intent(intent("P5", "approach", target="P1"))
    run(svc, apis, 0.3)
    apis["P1"].x += 4.0  # P1 moves away
    run(svc, apis, 0.3)
    moves = [c for c in apis["P5"].calls if c[0] == "move_to"]
    assert len(moves) >= 2


def test_the_walk_speed_comes_from_the_person(tmp_path):
    svc, _, apis = make(tmp_path)
    svc.step(DT)
    svc.on_intent(intent("P5", "walk_to", target="entrance"))
    run(svc, apis, 5.0)
    speeds = [c[1] for c in apis["P5"].calls if c[0] == "set_speed"]
    assert speeds == [pytest.approx(pe.clamp_speed(svc.world.people["P5"].inst.walk_speed), abs=0.01)]
    assert pe.clamp_speed(0.05) == pe.SPEED_RANGE[0] and pe.clamp_speed(9) == pe.SPEED_RANGE[1]


def test_people_without_a_body_are_still_walked_by_the_world(tmp_path):
    svc, bodies, apis = make(tmp_path, ("P5",))
    svc.step(DT)
    svc.on_intent(intent("P1", "stand_up"))
    svc.on_intent(intent("P1", "walk_to", target="entrance"))
    x0 = svc.world.people["P1"].x
    run(svc, apis, 5.0)
    assert svc.world.people["P1"].x != x0 or svc.world.people["P1"].y != svc.world.site.seats["B2"].y


def test_a_person_who_joins_later_gets_a_character_from_the_spawner(tmp_path):
    svc, bodies, apis = make(tmp_path)
    made = []

    def spawner(world, pid):
        if not pid.startswith("N"):
            return None
        made.append(pid)
        return pe.PersonController(pid, FakeApi(0.0, 0.0))

    bodies.spawner = spawner
    svc.step(DT)
    pid = svc.world.spawn("coughing_adult", svc.t)
    svc.step(DT)
    assert made == [pid] and pid in svc.world.body_driven and pid in bodies.controllers
    svc.step(DT)
    assert made == [pid]  # once


def test_without_a_spawner_a_joiner_stays_kinematic(tmp_path):
    svc, bodies, _ = make(tmp_path)
    svc.step(DT)
    pid = svc.world.spawn("coughing_adult", svc.t)
    svc.step(DT)
    assert pid not in svc.world.body_driven
