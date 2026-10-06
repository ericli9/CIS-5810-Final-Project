"""Runs the whole pipeline over the generated demo, headless."""

from pathlib import Path

import pytest

from smartcheckout.config import AppConfig
from smartcheckout.demo import build_demo
from smartcheckout.pipeline import Pipeline

CATALOG = Path(__file__).resolve().parents[1] / "configs" / "catalog.json"


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo")
    config_path = build_demo(root / "data", root / "demo.json", catalog_rel=str(CATALOG))
    cfg = AppConfig.load(config_path)
    pipeline = Pipeline(cfg, backend="scripted", alert_dir=root / "alerts")
    while True:
        result = pipeline.step()
        if result.finished:
            break
    report = pipeline.report()
    pipeline.close()
    return pipeline, report


def test_the_demo_feeds_and_config_are_written(demo_run):
    pipeline, report = demo_run
    assert {c["id"] for c in report["cameras"]} == {"bin", "bag"}
    assert report["frames"] >= 200


def test_bin_items_end_up_on_the_receipt(demo_run):
    pipeline, report = demo_run
    items = {item["display"]: item["qty"] for item in report["receipt"]["items"]}
    assert items == {"Bananas (bunch)": 1, "Bottled drink": 1, "Orange": 1}
    # banana 1.49 + bottle 2.49 + orange 0.79; the cup was lifted back out
    assert report["receipt"]["total"] == pytest.approx(4.77)
    assert report["receipt"]["items_added"] == 4
    assert report["receipt"]["items_removed"] == 1
    assert report["receipt"]["needs_attention"] is False


def test_bagged_items_that_went_through_the_bin_clear(demo_run):
    pipeline, report = demo_run
    cleared = {row["class"] for row in report["loss_prevention"]["cleared"]}
    assert cleared == {"banana", "bottle"}


def test_the_apple_that_skipped_the_bin_camera_raises_one_alert(demo_run):
    pipeline, report = demo_run
    alerts = report["loss_prevention"]["alerts"]
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["class"] == "apple"
    assert alert["camera"] == "bag"
    assert alert["zone"] == "shopping-bag"
    assert Path(alert["snapshot"]).exists()


def test_running_twice_from_a_reset_gives_the_same_answer(demo_run):
    pipeline, report = demo_run
    pipeline.reset()
    while True:
        if pipeline.step().finished:
            break
    again = pipeline.report()
    assert again["receipt"]["total"] == report["receipt"]["total"]
    assert len(again["loss_prevention"]["alerts"]) == 1
