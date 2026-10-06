from collections import Counter

import pytest

from smartcheckout.catalog import Catalog, Product
from smartcheckout.checkout import Cart
from smartcheckout.lossprev import ScanGate
from smartcheckout.zones import Zone
from smartcheckout.zonestate import TrackState, ZoneEvent


@pytest.fixture
def catalog() -> Catalog:
    return Catalog(
        {
            "apple": Product("apple", "P1", "Apple", 0.89),
            "bottle": Product("bottle", "P2", "Bottled drink", 2.49),
        }
    )


def zone(name: str, role: str) -> Zone:
    return Zone(name=name, role=role, polygon=[(0, 0), (1, 0), (1, 1)])


def event(kind: str, role: str, cls_name: str, track_id: int, camera="cam", frame=0) -> ZoneEvent:
    track = TrackState(track_id=track_id, cls_votes=Counter({cls_name: 5}))
    return ZoneEvent(kind=kind, camera=camera, zone=zone(role, role), track=track, frame_idx=frame)


# -- cart --------------------------------------------------------------------


def test_bin_entry_prices_the_item(catalog):
    cart = Cart(catalog)
    cart.handle(event("enter", "bin", "apple", 1))
    assert cart.total == pytest.approx(0.89)
    assert [(g.display, g.qty) for g in cart.groups()] == [("Apple", 1)]


def test_duplicate_entry_for_one_track_adds_one_line(catalog):
    cart = Cart(catalog)
    cart.handle(event("enter", "bin", "apple", 1))
    cart.handle(event("enter", "bin", "apple", 1))
    assert len(cart.lines) == 1


def test_identical_items_group_with_a_quantity(catalog):
    cart = Cart(catalog)
    cart.handle(event("enter", "bin", "apple", 1))
    cart.handle(event("enter", "bin", "apple", 2))
    groups = cart.groups()
    assert groups[0].qty == 2
    assert groups[0].subtotal == pytest.approx(1.78)


def test_taking_an_item_out_of_the_bin_removes_its_line(catalog):
    cart = Cart(catalog)
    cart.handle(event("enter", "bin", "bottle", 3))
    cart.handle(event("exit", "bin", "bottle", 3))
    assert cart.lines == {}
    assert cart.total == 0.0
    assert cart.removed == 1


def test_unknown_class_is_flagged_for_an_attendant(catalog):
    cart = Cart(catalog)
    cart.handle(event("enter", "bin", "elephant", 4))
    assert cart.needs_attention
    assert cart.total == 0.0
    assert cart.receipt()["items"][0]["recognized"] is False


def test_non_bin_zones_do_not_touch_the_cart(catalog):
    cart = Cart(catalog)
    cart.handle(event("enter", "bag", "apple", 1))
    cart.handle(event("enter", "scan", "apple", 2))
    assert cart.lines == {}


# -- scan gate ---------------------------------------------------------------


def test_bagging_after_a_scan_on_the_same_track_clears(catalog):
    gate = ScanGate(catalog, fps=10)
    assert gate.handle(event("enter", "scan", "apple", 1)) is None
    assert gate.handle(event("enter", "bag", "apple", 1)) is None
    assert gate.alerts == []
    assert gate.cleared[0]["why"].startswith("same track")


def test_bagging_without_any_scan_alerts(catalog):
    gate = ScanGate(catalog, fps=10)
    alert = gate.handle(event("enter", "bag", "apple", 1, frame=50))
    assert alert is not None
    assert alert.display == "Apple"
    assert alert.time_s == pytest.approx(5.0)
    assert "without passing a scan zone" in alert.reason


def test_one_alert_per_track(catalog):
    gate = ScanGate(catalog, fps=10)
    gate.handle(event("enter", "bag", "apple", 1))
    gate.handle(event("enter", "bag", "apple", 1))
    assert len(gate.alerts) == 1


def test_a_credit_from_another_camera_clears_the_bag(catalog):
    gate = ScanGate(catalog, fps=10)
    gate.handle(event("enter", "bin", "bottle", 1, camera="cam-bin"))
    assert gate.handle(event("enter", "bag", "bottle", 9, camera="cam-bag")) is None
    assert gate.alerts == []
    assert gate.credits["bottle"] == 0  # the credit was consumed, not reusable


def test_a_credit_clears_only_one_item(catalog):
    gate = ScanGate(catalog, fps=10)
    gate.handle(event("enter", "bin", "bottle", 1, camera="cam-bin"))
    gate.handle(event("enter", "bag", "bottle", 9, camera="cam-bag"))
    alert = gate.handle(event("enter", "bag", "bottle", 10, camera="cam-bag"))
    assert alert is not None
    assert len(gate.alerts) == 1


def test_a_credit_does_not_cover_a_different_class(catalog):
    gate = ScanGate(catalog, fps=10)
    gate.handle(event("enter", "bin", "bottle", 1, camera="cam-bin"))
    assert gate.handle(event("enter", "bag", "apple", 9, camera="cam-bag")) is not None


def test_pulling_an_item_back_out_of_the_bin_revokes_its_credit(catalog):
    gate = ScanGate(catalog, fps=10)
    gate.handle(event("enter", "bin", "apple", 1, camera="cam-bin"))
    gate.handle(event("exit", "bin", "apple", 1, camera="cam-bin"))
    assert gate.credits["apple"] == 0
    assert gate.handle(event("enter", "bag", "apple", 9, camera="cam-bag")) is not None


def test_leaving_a_scan_zone_keeps_the_credit(catalog):
    gate = ScanGate(catalog, fps=10)
    gate.handle(event("enter", "scan", "apple", 1, camera="cam-a"))
    gate.handle(event("exit", "scan", "apple", 1, camera="cam-a"))
    assert gate.handle(event("enter", "bag", "apple", 9, camera="cam-b")) is None
