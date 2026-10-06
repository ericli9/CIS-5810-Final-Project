from smartcheckout.detect import Detection
from smartcheckout.zones import Zone, ZoneSet
from smartcheckout.zonestate import ZoneOccupancy

W, H = 200, 100


def square(name="z", role="bin", x0=0.0, y0=0.0, x1=0.5, y1=1.0) -> Zone:
    zone = Zone(name=name, role=role, polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    zone.bind(W, H)
    return zone


def test_coverage_fully_inside_and_outside():
    zone = square()  # left half of the frame
    assert zone.coverage((10, 10, 50, 50)) == 1.0
    assert zone.coverage((120, 10, 160, 50)) == 0.0


def test_coverage_half_overlap():
    zone = square()
    # box straddles x=100, the zone's right edge; the shared edge column is
    # rasterized as inside, so expect half plus one column out of forty
    assert abs(zone.coverage((80, 20, 120, 60)) - 0.5) < 0.03


def test_coverage_of_concave_zone_excludes_the_notch():
    # a "C" shape: the middle-right of the box is cut out
    zone = Zone(
        name="c",
        role="bin",
        polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 0.25), (0.5, 0.25), (0.5, 0.75), (1.0, 0.75), (1.0, 1.0), (0.0, 1.0)],
    )
    zone.bind(W, H)
    assert zone.coverage((120, 30, 180, 70)) == 0.0
    assert zone.coverage((10, 30, 70, 70)) == 1.0


def test_zone_requires_known_role():
    try:
        Zone(name="z", role="freezer", polygon=[(0, 0), (1, 0), (1, 1)])
    except ValueError as exc:
        assert "role" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unknown role")


def test_occupancy_debounces_entry_and_exit():
    zones = ZoneSet([square()])
    zones.bind(W, H)
    occ = ZoneOccupancy("cam", zones, coverage_threshold=0.3, confirm_frames=3, exit_frames=2)

    inside = Detection(cls_name="apple", conf=0.9, box=(10, 10, 50, 50), track_id=1)
    assert occ.update([inside], 0) == []
    assert occ.update([inside], 1) == []
    events = occ.update([inside], 2)
    assert [(e.kind, e.zone.name) for e in events] == [("enter", "z")]
    assert occ.update([inside], 3) == []  # no repeat while it stays

    outside = Detection(cls_name="apple", conf=0.9, box=(150, 10, 190, 50), track_id=1)
    assert occ.update([outside], 4) == []
    events = occ.update([outside], 5)
    assert [(e.kind, e.zone.name) for e in events] == [("exit", "z")]


def test_occupancy_releases_zones_after_a_track_is_lost():
    zones = ZoneSet([square()])
    zones.bind(W, H)
    occ = ZoneOccupancy("cam", zones, confirm_frames=1, lost_frames=3)
    det = Detection(cls_name="apple", conf=0.9, box=(10, 10, 50, 50), track_id=7)
    assert len(occ.update([det], 0)) == 1
    assert occ.update([], 1) == []
    assert occ.update([], 2) == []
    events = occ.update([], 3)
    assert [(e.kind, e.zone.name) for e in events] == [("exit", "z")]
    assert occ.tracks == {}


def test_untracked_detections_never_change_state():
    zones = ZoneSet([square()])
    zones.bind(W, H)
    occ = ZoneOccupancy("cam", zones, confirm_frames=1)
    det = Detection(cls_name="apple", conf=0.9, box=(10, 10, 50, 50), track_id=None)
    assert occ.update([det], 0) == []
    assert occ.tracks == {}


def test_class_label_is_a_majority_vote_over_the_track():
    zones = ZoneSet([square()])
    zones.bind(W, H)
    occ = ZoneOccupancy("cam", zones, confirm_frames=4)
    box = (10, 10, 50, 50)
    occ.update([Detection("apple", 0.9, box, 1)], 0)
    occ.update([Detection("orange", 0.5, box, 1)], 1)  # one bad frame
    occ.update([Detection("apple", 0.9, box, 1)], 2)
    events = occ.update([Detection("apple", 0.9, box, 1)], 3)
    assert events[0].track.cls_name == "apple"
