"""Isaac's world-frame, foot-centred Detection3DArray with numeric class ids becomes WardZero's class-only bbox_3d
with the world's anonymous tracks. No Isaac, no ROS."""
import json
import math

from wardzero_agents.robots.bbox_adapter import BboxAdapter, detections_from_msg, nearest_person_resolver

LABELS = json.dumps({"0": {"class": "person"}, "1": {"class": "bed"}})


def det(cls, x, y, z, w, size=(0.52, 0.47, 1.60), qz=0.0, qw=1.0):
    return {"class_id": cls, "center": (x, y, z), "orientation": (0.0, 0.0, qz, qw), "size": size}


def stamp(t):
    return {"sec": int(t), "nanosec": int(round((t - int(t)) * 1e9))}


class FakeWorld:
    """What the resolver needs: people with positions, and tracks handed out in the order seen."""

    def __init__(self, people):
        self.people = people
        self.order = []

    def track(self, pid):
        if pid not in self.order:
            self.order.append(pid)
        return f"t{self.order.index(pid) + 1}"


def adapter(people):
    w = FakeWorld(people)
    return BboxAdapter(nearest_person_resolver(lambda: w.people, w.track, gate_m=0.8)), w


def test_foot_centred_world_box_becomes_a_geometric_centre_with_a_track():
    ad, _ = adapter({"P1": (1.0, 0.2), "P2": (4.0, 4.0)})
    ad.set_labels(LABELS)
    out = ad.convert(1.5, [det("0", 1.02, 0.18, -0.001, 0, qz=0.7071068, qw=0.7071068)])
    assert out["header"] == {"stamp": 1.5, "frame_id": "map"}
    (d,) = out["detections"]
    assert d["id"] == "t1" and d["results"] == [{"class_id": "person", "score": 1.0}]
    c = d["bbox"]["center"]
    assert (c["x"], c["y"]) == (1.02, 0.18)
    assert c["z"] == 0.799  # -0.001 (the feet) + 1.60 / 2
    assert c["yaw"] == round(math.pi / 2, 3)
    assert d["bbox"]["size"] == {"x": 0.52, "y": 0.47, "z": 1.6}


def test_numeric_class_ids_are_mapped_and_unknown_classes_are_dropped():
    ad, _ = adapter({"P1": (1.0, 0.2)})
    ad.set_labels(LABELS)
    out = ad.convert(0.0, [det("1", 1.0, 0.2, 0, 0), det("7", 1.0, 0.2, 0, 0)])
    assert out["detections"] == []  # a bed is not a person; id 7 is not in the labels
    assert (ad.unknown_class, ad.unmatched) == (2, 0)
    fresh, _ = adapter({"P1": (1.0, 0.2)})
    assert fresh.convert(0.0, [det("0", 1.0, 0.2, 0, 0)])["detections"] == []  # no labels yet: nothing is called a person


def test_labels_message_shapes():
    ad, _ = adapter({"P1": (1.0, 0.2)})
    ad.set_labels('{"data": {"0": {"class": "person"}}}')  # the camera helper wraps the map in "data" in some builds
    assert ad.convert(0.0, [det("0", 1.0, 0.2, 0, 0)])["detections"][0]["id"] == "t1"
    ad.set_labels("not json")  # a bad message keeps the labels already known
    assert ad.convert(0.0, [det("0", 1.0, 0.2, 0, 0)])["detections"][0]["id"] == "t1"
    ad.set_labels('{"0": {"class": "person, adult"}}')  # a comma list of semantics: person counts
    assert ad.convert(0.0, [det("0", 1.0, 0.2, 0, 0)])["detections"][0]["id"] == "t1"


def test_tracks_are_stable_and_follow_the_person_not_the_order():
    ad, w = adapter({"P1": (1.0, 0.2), "P2": (3.0, 0.2)})
    ad.set_labels(LABELS)
    a = ad.convert(0.0, [det("0", 3.0, 0.2, 0, 0), det("0", 1.0, 0.2, 0, 0)])
    ids = {(d["bbox"]["center"]["x"]): d["id"] for d in a["detections"]}
    assert ids == {3.0: "t1", 1.0: "t2"}  # numbered in the order the world first saw them
    w.people["P1"], w.people["P2"] = (1.3, 0.4), (3.2, 0.1)  # both walked
    b = ad.convert(0.5, [det("0", 1.3, 0.4, 0, 0), det("0", 3.2, 0.1, 0, 0)])
    assert {d["bbox"]["center"]["x"]: d["id"] for d in b["detections"]} == {1.3: "t2", 3.2: "t1"}


def test_one_person_gets_one_detection_and_a_stranger_is_not_given_a_track():
    ad, _ = adapter({"P1": (1.0, 0.2)})
    ad.set_labels(LABELS)
    out = ad.convert(0.0, [det("0", 1.0, 0.2, 0, 0), det("0", 1.1, 0.2, 0, 0), det("0", 9.0, 9.0, 0, 0)])
    assert [d["bbox"]["center"]["x"] for d in out["detections"]] == [1.0]  # the nearer box wins P1
    assert (ad.unmatched, ad.unknown_class) == (2, 0)


def test_the_message_never_carries_a_person_id():
    ad, _ = adapter({"P1": (1.0, 0.2), "SB3": (2.0, 0.2)})
    ad.set_labels(LABELS)
    text = json.dumps(ad.convert(0.0, [det("0", 1.0, 0.2, 0, 0), det("0", 2.0, 0.2, 0, 0)]))
    assert "P1" not in text and "SB3" not in text


def test_a_sitting_and_a_lying_box_keep_their_size():
    ad, _ = adapter({"P1": (1.0, 0.2), "P2": (3.0, 0.2)})
    ad.set_labels(LABELS)
    out = ad.convert(0.0, [det("0", 1.0, 0.2, 0.0, 0, size=(0.6, 0.6, 1.2)), det("0", 3.0, 0.2, 0.9, 0, size=(1.8, 0.6, 0.5))])
    assert [d["bbox"]["size"]["z"] for d in out["detections"]] == [1.2, 0.5]
    assert [d["bbox"]["center"]["z"] for d in out["detections"]] == [0.6, 1.15]


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def ros_det(cls, x, y, z, qz, qw, size):
    pose = _NS(position=_NS(x=x, y=y, z=z), orientation=_NS(x=0.0, y=0.0, z=qz, w=qw))
    return _NS(results=[_NS(hypothesis=_NS(class_id=cls, score=0.9))],
               bbox=_NS(center=_NS(position=pose.position, orientation=pose.orientation), size=_NS(x=size[0], y=size[1], z=size[2])))


def test_detections_from_a_ros_message():
    msg = _NS(header=_NS(stamp=_NS(**stamp(12.25))), detections=[ros_det("0", 1.0, 0.2, 0.0, 0.0, 1.0, (0.5, 0.5, 1.7))])
    t, dets = detections_from_msg(msg)
    assert t == 12.25
    assert dets == [{"class_id": "0", "center": (1.0, 0.2, 0.0), "orientation": (0.0, 0.0, 0.0, 1.0), "size": (0.5, 0.5, 1.7)}]
    empty = _NS(header=msg.header, detections=[_NS(results=[], bbox=msg.detections[0].bbox)])
    assert detections_from_msg(empty)[1][0]["class_id"] == ""  # a box with no hypothesis has no class
