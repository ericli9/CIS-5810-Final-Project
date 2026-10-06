"""Serves the web console: MJPEG camera streams plus a JSON state feed.

Standard library only -- no web framework. One worker thread owns the pipeline
(which is not thread safe); request handlers only ever read the snapshot it
publishes under a lock, and send control actions back through flags the worker
applies between frames.
"""

from __future__ import annotations

import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from .pipeline import Pipeline

WEB_ROOT = Path(__file__).resolve().parent / "web"
BOUNDARY = "frameboundary"
JPEG_QUALITY = 80
MAX_TIMELINE = 60


class ConsoleState:
    """The latest frames and JSON snapshot, shared worker -> request threads."""

    def __init__(self) -> None:
        self.lock = threading.Condition()
        self.seq = 0
        self.jpegs: dict[str, bytes] = {}
        self.snapshot: dict = {"status": "starting"}

    def publish(self, jpegs: dict[str, bytes], snapshot: dict) -> None:
        with self.lock:
            self.jpegs = jpegs
            self.snapshot = snapshot
            self.seq += 1
            self.lock.notify_all()

    def wait_for(self, last_seq: int, timeout: float = 5.0) -> tuple[int, dict[str, bytes]]:
        with self.lock:
            if self.seq == last_seq:
                self.lock.wait(timeout)
            return self.seq, self.jpegs

    def read(self) -> dict:
        with self.lock:
            return self.snapshot


class PipelineWorker(threading.Thread):
    def __init__(self, pipeline: Pipeline, speed: float = 1.0):
        super().__init__(name="pipeline", daemon=True)
        self.pipeline = pipeline
        pipeline.draw_header = False  # the console draws the camera's name in HTML
        self.speed = speed
        self.state = ConsoleState()
        self.paused = False
        self._restart = False
        self._stop = threading.Event()
        self._finished = False
        self._session = 0  # bumped on each restart, so a run can be told from the next

    # -- controls, called from request threads --------------------------------

    def pause(self, value: bool) -> None:
        self.paused = value

    def restart(self) -> None:
        self._restart = True

    def shutdown(self) -> None:
        self._stop.set()

    # -- worker ---------------------------------------------------------------

    def run(self) -> None:
        period = 1.0 / (self.pipeline.fps_nominal * self.speed) if self.speed > 0 else 0.0
        self._publish(None)
        while not self._stop.is_set():
            started = time.perf_counter()

            if self._restart:
                self._restart = False
                self._finished = False
                self.pipeline.reset()
                self.paused = False
                self._session += 1

            if self.paused or self._finished:
                time.sleep(0.05)
                self._publish(None)
                continue

            result = self.pipeline.step()
            if result.finished:
                self._finished = True
            self._publish(result.frames)

            if period:
                time.sleep(max(0.0, period - (time.perf_counter() - started)))

    def _publish(self, frames: list | None) -> None:
        jpegs = self.state.jpegs
        if frames:
            jpegs = {}
            for cam, frame in zip(self.pipeline.cameras, frames):
                ok, buf = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                )
                if ok:
                    jpegs[cam.cfg.id] = buf.tobytes()
        self.state.publish(jpegs, self._snapshot())

    def _snapshot(self) -> dict:
        pipeline = self.pipeline
        fps = pipeline.fps_nominal
        gate = pipeline.gate
        status = "paused" if self.paused else ("ended" if self._finished else "running")
        last_alert = gate.alerts[-1] if gate.alerts else None
        alert_active = bool(
            last_alert and pipeline.frame_idx - last_alert.frame_idx <= fps * 2.5
        )
        timeline = [e.to_dict(fps) for e in pipeline.timeline()[-MAX_TIMELINE:]]
        return {
            "status": status,
            "session": self._session,
            "frame": pipeline.frame_idx,
            "time_s": round(pipeline.frame_idx / fps, 1) if fps else 0.0,
            "fps": round(pipeline.stats()["fps"], 1),
            "backend": pipeline.backend_label,
            "currency": pipeline.catalog.currency,
            "cameras": [
                {
                    "id": cam.cfg.id,
                    "label": cam.cfg.label or cam.cfg.id,
                    "roles": sorted({z.role for z in cam.cfg.zones}),
                    "width": cam.size[0],
                    "height": cam.size[1],
                    "alerting": bool(
                        alert_active and last_alert and last_alert.camera == cam.cfg.id
                    ),
                }
                for cam in pipeline.cameras
            ],
            "receipt": pipeline.cart.receipt(),
            "gate": {
                "accounted": len(gate.accounted_tracks),
                "cleared": len(gate.cleared),
                "credits": sum(v for v in gate.credits.values() if v > 0),
                "alerts": len(gate.alerts),
                "alert_active": alert_active,
                "last_alert": last_alert.to_dict() if last_alert else None,
            },
            "timeline": list(reversed(timeline)),  # newest first, as the log reads
        }


class ConsoleHandler(BaseHTTPRequestHandler):
    worker: PipelineWorker  # set on the server class below
    server_version = "smartcheckout"

    def log_message(self, fmt: str, *args) -> None:  # quieter than the default
        pass

    # -- helpers --------------------------------------------------------------

    def _send(self, body: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        self._send(json.dumps(payload).encode("utf-8"), "application/json", status)

    # -- routes ---------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802  (BaseHTTPRequestHandler's spelling)
        path = self.path.split("?", 1)[0]
        try:
            if path in ("/", "/index.html", "/console.html"):
                self._send((WEB_ROOT / "console.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(self.worker.state.read())
            elif path.startswith("/api/stream/"):
                self._stream(path.rsplit("/", 1)[-1])
            elif path.startswith("/api/frame/"):
                self._frame(path.rsplit("/", 1)[-1].removesuffix(".jpg"))
            else:
                self._json({"error": "not found", "path": path}, HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # the browser closed the tab or navigated away mid-stream

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/api/control":
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "body must be JSON"}, HTTPStatus.BAD_REQUEST)
            return

        action = payload.get("action")
        if action == "pause":
            self.worker.pause(True)
        elif action == "resume":
            self.worker.pause(False)
        elif action == "restart":
            self.worker.restart()
        else:
            self._json(
                {"error": f"unknown action {action!r}", "actions": ["pause", "resume", "restart"]},
                HTTPStatus.BAD_REQUEST,
            )
            return
        self._json({"ok": True, "action": action})

    def _frame(self, camera_id: str) -> None:
        jpeg = self.worker.state.jpegs.get(camera_id)
        if jpeg is None:
            self._json({"error": f"no camera {camera_id!r}"}, HTTPStatus.NOT_FOUND)
            return
        self._send(jpeg, "image/jpeg")

    def _stream(self, camera_id: str) -> None:
        if camera_id not in {cam.cfg.id for cam in self.worker.pipeline.cameras}:
            self._json({"error": f"no camera {camera_id!r}"}, HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        last_seq = -1
        while True:
            last_seq, jpegs = self.worker.state.wait_for(last_seq)
            jpeg = jpegs.get(camera_id)
            if jpeg is None:
                continue
            self.wfile.write(
                f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
            )
            self.wfile.write(jpeg)
            self.wfile.write(b"\r\n")


def serve(
    pipeline: Pipeline,
    host: str = "127.0.0.1",
    port: int = 8000,
    speed: float = 1.0,
) -> tuple[ThreadingHTTPServer, PipelineWorker]:
    """Start the worker and an HTTP server. Returns both; caller runs/closes them."""
    worker = PipelineWorker(pipeline, speed=speed)
    handler = type("BoundConsoleHandler", (ConsoleHandler,), {"worker": worker})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    worker.start()
    return httpd, worker
