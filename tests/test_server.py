"""Smoke-tests the web console's HTTP surface against a live pipeline."""

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from smartcheckout.config import AppConfig
from smartcheckout.demo import build_demo
from smartcheckout.pipeline import Pipeline
from smartcheckout.server import serve

CATALOG = Path(__file__).resolve().parents[1] / "configs" / "catalog.json"


def get(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, response.headers, response.read()


def post(url: str, payload: dict, timeout: float = 10.0):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read())


@pytest.fixture(scope="module")
def console(tmp_path_factory):
    root = tmp_path_factory.mktemp("console")
    config_path = build_demo(root / "data", root / "demo.json", catalog_rel=str(CATALOG))
    pipeline = Pipeline(AppConfig.load(config_path), backend="scripted")
    httpd, worker = serve(pipeline, host="127.0.0.1", port=0, speed=0)
    import threading

    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield base, worker, pipeline
    finally:
        worker.shutdown()
        httpd.shutdown()
        httpd.server_close()
        pipeline.close()


def wait_until(predicate, tries: int = 400, delay: float = 0.05):
    import time

    for _ in range(tries):
        if predicate():
            return True
        time.sleep(delay)
    return False


def test_the_console_page_is_served(console):
    base, _, _ = console
    status, headers, body = get(f"{base}/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"<title>Lane console" in body
    assert b"/api/stream/" in body  # the page knows where the feeds live


def test_state_reports_the_cameras_and_an_empty_cart_at_first(console):
    base, _, _ = console
    _, _, body = get(f"{base}/api/state")
    state = json.loads(body)
    assert state["status"] in {"starting", "running", "paused", "ended"}
    assert [c["id"] for c in state["cameras"]] == ["bin", "bag"]
    assert set(state["gate"]) >= {"alerts", "cleared", "accounted", "credits", "alert_active"}
    assert "items" in state["receipt"]


def test_state_fills_in_as_the_session_runs(console):
    base, _, pipeline = console
    assert wait_until(lambda: pipeline.gate.alerts), "the demo never reached its alert"
    state = json.loads(get(f"{base}/api/state")[2])
    assert state["receipt"]["total"] == pytest.approx(4.77)
    assert state["gate"]["alerts"] == 1
    assert state["timeline"], "the decision log should not be empty"
    # newest first, which is the order the console renders
    frames = [entry["frame"] for entry in state["timeline"]]
    assert frames == sorted(frames, reverse=True)


def test_a_single_frame_is_served_as_jpeg(console):
    base, _, _ = console
    status, headers, body = get(f"{base}/api/frame/bin.jpg")
    assert status == 200
    assert headers["Content-Type"] == "image/jpeg"
    assert body[:2] == b"\xff\xd8"  # JPEG start-of-image marker


def test_pause_and_resume_round_trip(console):
    base, worker, _ = console
    assert post(f"{base}/api/control", {"action": "pause"})[1]["ok"] is True
    assert wait_until(lambda: json.loads(get(f"{base}/api/state")[2])["status"] == "paused")
    post(f"{base}/api/control", {"action": "resume"})
    assert wait_until(lambda: json.loads(get(f"{base}/api/state")[2])["status"] != "paused")


def test_restart_starts_a_new_session(console):
    base, _, _ = console
    before = json.loads(get(f"{base}/api/state")[2])["session"]
    post(f"{base}/api/control", {"action": "restart"})
    assert wait_until(lambda: json.loads(get(f"{base}/api/state")[2])["session"] == before + 1)


def test_unknown_routes_and_actions_are_rejected(console):
    base, _, _ = console
    with pytest.raises(urllib.error.HTTPError) as caught:
        get(f"{base}/api/nope")
    assert caught.value.code == 404

    with pytest.raises(urllib.error.HTTPError) as caught:
        post(f"{base}/api/control", {"action": "selfdestruct"})
    assert caught.value.code == 400
