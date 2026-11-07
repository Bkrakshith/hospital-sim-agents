"""A person whose walk out of a seat fails is never left standing inside the seat rows (re-QA bug 11).

In Isaac ae_waiting George stood up from seat A2, stood inside the seat row (off the NavMesh), every MoveTo to
AE_Triage failed at once ("walk ... blocked after 3 re-issues") and he stood in the row for 63 s. The fake Behavior
API here models that: a MoveTo that starts inside furniture fails at once."""
import math

from wardzero_agents.isaac_host import people as pe
from tests.test_isaac_people import DT, FakeApi, intent, make, run, status


class OffMeshApi(FakeApi):
    """MoveTo fails at once while the character stands inside furniture (off the NavMesh), as IRA's does."""

    blocked = None  # set to the site grid's blocked(x, y) by the test

    def move_to(self, x, y):
        tid = super().move_to(x, y)
        if self.blocked is not None and self.blocked(self.x, self.y):
            self.tasks[tid]["fail"] = True
        return tid


class TeleportApi(OffMeshApi):
    def teleport(self, x, y, yaw):
        self.calls.append(("teleport", round(x, 2), round(y, 2)))
        self.x, self.y, self.yaw = x, y, yaw
        self.action = "Idle"


def _make(tmp_path, cls):
    svc, bodies, apis = make(tmp_path)
    old = apis["P5"]
    api = cls(old.x, old.y, old.yaw, seats=old.seats)
    api.blocked = svc.world.site.grid.blocked
    ctrl = bodies.controllers["P5"]
    ctrl.api = api
    bodies.people["P5"] = api.pose  # the world reads the new character's pose back
    apis["P5"] = api
    svc.step(DT)
    return svc, apis


def _in_furniture(svc, pid):
    p = svc.world.people[pid]
    return p.posture in ("standing", "walking") and svc.world.site.grid.blocked(p.x, p.y)


def test_the_seat_rows_are_furniture_on_the_site_grid(tmp_path):
    svc, _ = _make(tmp_path, OffMeshApi)
    seat = svc.world.site.seats["D2"]
    assert svc.world.site.grid.blocked(seat.x, seat.y)
    assert not svc.world.site.grid.blocked(*pe.seat_approach(seat))


def test_a_walk_that_cannot_start_in_the_seat_row_steps_out_in_front_of_the_seat_and_goes(tmp_path):
    svc, apis = _make(tmp_path, TeleportApi)
    svc.on_intent(intent("P5", "walk_to", target="entrance"))
    run(svc, apis, 90.0)
    st = status(svc, "P5")
    assert st.state == "done", (st.state, st.reason)
    tel = [c for c in apis["P5"].calls if c[0] == "teleport"]
    assert len(tel) == 1
    front = pe.seat_approach(svc.world.site.seats["D2"])
    assert math.hypot(tel[0][1] - front[0], tel[0][2] - front[1]) < 0.6  # a step out of the row, to the floor in front


def test_a_walk_out_of_a_seat_that_fails_for_good_puts_the_person_back_in_the_seat(tmp_path):
    svc, apis = _make(tmp_path, OffMeshApi)  # no way to step out: the character can only sit back down
    svc.on_intent(intent("P5", "walk_to", target="entrance"))
    stood = False
    for _ in range(int(60 / DT)):
        for a in apis.values():
            a.tick(DT)
        svc.step(DT)
        stood = stood or svc.world.people["P5"].posture in ("standing", "walking")
    st = status(svc, "P5")
    assert st.state == "failed" and st.reason == "blocked"
    assert stood
    assert st.posture == "seated" and svc.world.people["P5"].seat == "D2"
    assert apis["P5"].calls[-1][0] == "sit" or ("sit", "/World/Seats/D2") in apis["P5"].calls[2:]
    assert not _in_furniture(svc, "P5")


def test_the_seat_taken_meanwhile_snaps_the_person_to_the_nearest_free_floor(tmp_path):
    svc, apis = _make(tmp_path, TeleportApi)
    apis["P5"].fail_walks = 99  # every walk fails, even from the free floor
    svc.on_intent(intent("P5", "walk_to", target="entrance"))
    run(svc, apis, 2.5)  # standing now, the seat released
    other = next(pid for pid in svc.world.people if pid != "P5" and svc.world.people[pid].posture != "gone")
    svc.world.people[other].seat = "D2"
    run(svc, apis, 90.0)
    st = status(svc, "P5")
    assert st.state == "failed" and st.reason == "blocked"
    assert st.posture == "standing" and not _in_furniture(svc, "P5")

