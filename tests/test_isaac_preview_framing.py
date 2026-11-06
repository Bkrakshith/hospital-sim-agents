"""The overhead/preview camera frames the kept rooms tightly in the 16:9 frame (pure: no Kit).

The camera is projected here the way ``recording.look_at`` (Gf ``SetLookAt`` with +z up) places it: forward from the eye
to the target, image right = forward x up, image up = right x forward."""
import pytest

from wardzero_agents.isaac_host import recording as R

ASPECT = 16 / 9
# the kept rooms of the meal_round preset (the ward: bays B1-B6 along Ward_Corridor, Ward_Entrance and two side rooms)
MEAL_ROUND = [(20.87, 25.2, 26.75, 28.58), (20.86, 17.81, 26.75, 21.19), (20.87, 14.12, 26.75, 17.49),
              (20.86, 6.73, 26.75, 10.07), (20.86, 3.01, 26.75, 6.37), (11.27, 21.51, 17.16, 24.89),
              (17.2, 4.0, 20.8, 32.0), (11.27, 4.15, 20.76, 10.25), (20.86, 10.41, 22.15, 13.79),
              (22.25, 10.41, 26.75, 12.05)]


def _sub(a, b):
    return tuple(u - v for u, v in zip(a, b))


def _dot(a, b):
    return sum(u * v for u, v in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _unit(a):
    n = _dot(a, a) ** 0.5
    return tuple(v / n for v in a)


def project(v: R.OverheadView, p, aspect=ASPECT):
    """A floor point to normalised image coordinates: (-1..1 left to right, -1..1 bottom to top) inside the frame."""
    f = _unit(_sub(v.target, v.eye))
    right = _unit(_cross(f, (0.0, 0.0, 1.0)))
    up = _cross(right, f)
    d = _sub(p, v.eye)
    depth = _dot(d, f)
    half_w = v.aperture / 2 / v.focal  # tan of the half field of view, across
    return _dot(d, right) / depth / half_w, _dot(d, up) / depth / (half_w / aspect)


def bbox_in_image(v, rects, aspect=ASPECT, z=0.0):
    """(width, height) of the rooms' bounding box at height z as a fraction of the image, after checking that every
    corner of the box, from the floor up to the wall tops the near plane leaves, is inside the image."""
    x0, y0 = min(r[0] for r in rects), min(r[1] for r in rects)
    x1, y1 = max(r[2] for r in rects), max(r[3] for r in rects)
    for zz in (0.0, R.CEILING_CUT_M):
        for x in (x0, x1):
            for y in (y0, y1):
                u, w = project(v, (x, y, zz), aspect)
                assert -1.0 <= u <= 1.0 and -1.0 <= w <= 1.0, (x, y, zz, u, w)  # nothing of the rooms is cut off
    pts = [project(v, (x, y, z), aspect) for x in (x0, x1) for y in (y0, y1)]
    us, ws = [p[0] for p in pts], [p[1] for p in pts]
    return (max(us) - min(us)) / 2, (max(ws) - min(ws)) / 2


def test_a_ward_taller_than_wide_turns_so_it_fills_the_frame():
    v = R.overhead_view(MEAL_ROUND, ASPECT)
    assert v.rotated
    fw, fh = bbox_in_image(v, MEAL_ROUND, z=R.CEILING_CUT_M)
    assert max(fw, fh) == pytest.approx(1 / R.OVERHEAD_MARGIN, rel=1e-3)  # tight: only the margin on the long side
    fw, fh = bbox_in_image(v, MEAL_ROUND)  # the floor
    assert min(fw, fh) >= 0.75 and fw * fh >= 0.65  # the ward covers most of the frame, not a thin strip
    # unturned, the same ward was a strip about a third of the frame's width
    x0, x1 = min(r[0] for r in MEAL_ROUND), max(r[2] for r in MEAL_ROUND)
    y0, y1 = min(r[1] for r in MEAL_ROUND), max(r[3] for r in MEAL_ROUND)
    assert (x1 - x0) / ((y1 - y0) * ASPECT) < 0.35


def test_turned_camera_puts_the_long_floor_axis_across_the_image():
    v = R.overhead_view([(0.0, 0.0, 4.0, 30.0)], ASPECT)
    assert v.rotated
    left, right = project(v, (2.0, 0.0, 0.0)), project(v, (2.0, 30.0, 0.0))
    assert right[0] - left[0] > 1.6 and abs(right[1] - left[1]) < 1e-3  # y runs left to right across the frame
    assert v.eye[2] > v.near > 0 and v.near == pytest.approx(v.eye[2] - R.CEILING_CUT_M)


def test_wide_area_is_not_turned_and_fills_the_width():
    rects = [(-12.0, -3.5, 11.2, 9.0), (11.2, -4.5, 27.5, 29.0)]  # wider than tall in 16:9 terms
    v = R.overhead_view(rects, ASPECT)
    assert not v.rotated
    fw, fh = bbox_in_image(v, rects, z=R.CEILING_CUT_M)
    assert max(fw, fh) == pytest.approx(1 / R.OVERHEAD_MARGIN, rel=1e-3)


def test_a_wide_corridor_fills_the_width():
    rects = [(-35.5, 4.15, 11.21, 14.45)]
    v = R.overhead_view(rects, ASPECT)
    assert not v.rotated
    fw, _ = bbox_in_image(v, rects, z=R.CEILING_CUT_M)
    assert fw == pytest.approx(1 / R.OVERHEAD_MARGIN, rel=1e-3)
    assert bbox_in_image(v, rects)[0] > 0.85  # the floor too


def test_nearly_square_area_stays_unturned():
    for w, h in ((10.0, 10.0), (10.0, 6.0), (6.0, 10.0)):
        v = R.overhead_view([(0.0, 0.0, w, h)], ASPECT)
        bbox_in_image(v, [(0.0, 0.0, w, h)])
    assert not R.overhead_view([(0.0, 0.0, 10.0, 6.0)], ASPECT).rotated


@pytest.mark.parametrize("w,h", [(15.5, 29.0), (62.0, 29.0), (4.0, 30.0), (30.0, 4.0), (6.0, 4.0), (1.0, 1.0)])
def test_every_shape_is_fully_inside_the_frame(w, h):
    rects = [(5.0, -3.0, 5.0 + w, -3.0 + h)]
    v = R.overhead_view(rects, ASPECT)
    fw, fh = bbox_in_image(v, rects, z=R.CEILING_CUT_M)
    if v.eye[2] > R.CEILING_CUT_M + 1.0 + 1e-9:  # not held up by the ceiling cut: tight on one side
        assert max(fw, fh) == pytest.approx(1 / R.OVERHEAD_MARGIN, rel=1e-3)
