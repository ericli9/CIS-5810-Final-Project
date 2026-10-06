"""Per-frame orchestration across every camera."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .catalog import Catalog
from .checkout import Cart
from .config import AppConfig, CameraConfig
from .detect import Detection, ScriptedDetector, YoloDetector
from .lossprev import Alert, ScanGate
from .viz import draw_alert_banner, draw_camera_header, draw_detections, draw_zones
from .zonestate import ZoneEvent, ZoneOccupancy

BLANK_SOURCES = {"blank", "none", ""}


class CameraRuntime:
    """One camera: its capture, its detector, its zone state machine."""

    def __init__(self, cfg: CameraConfig, app: AppConfig, catalog: Catalog, backend: str):
        self.cfg = cfg
        self.app = app
        self.backend = backend
        self.frame_idx = 0
        self.finished = False
        self.last_frame: np.ndarray | None = None
        self.detections: list[Detection] = []
        self.cap: cv2.VideoCapture | None = None
        self.size = (640, 360)
        self.fps = 20.0

        if backend == "scripted":
            if cfg.detections is None:
                raise ValueError(f"camera {cfg.id!r}: scripted backend needs a 'detections' file")
            scripted = ScriptedDetector(cfg.detections)
            self.detector = scripted
            self.n_scripted = scripted.n_frames
            if scripted.width and scripted.height:
                self.size = (scripted.width, scripted.height)
            self.fps = scripted.fps or 20.0
        else:
            self.detector = YoloDetector(
                model_name=app.model,
                allowed_classes=catalog.known_classes(),
                min_conf=app.min_conf,
                imgsz=app.imgsz,
                device=app.device,
                tracker=app.tracker,
            )
            self.n_scripted = None

        self._open_capture()
        self.occupancy = ZoneOccupancy(
            camera=cfg.id,
            zoneset=cfg.zones,
            coverage_threshold=app.coverage_threshold,
            confirm_frames=app.confirm_frames,
            exit_frames=app.exit_frames,
            lost_frames=app.lost_frames,
        )
        cfg.zones.bind(*self.size)

    # -- capture -------------------------------------------------------------

    def _open_capture(self) -> None:
        source = self.cfg.capture_source
        if isinstance(source, str) and source.lower() in BLANK_SOURCES:
            self.cap = None
            return
        if isinstance(source, str) and not source.startswith(("http", "rtsp")):
            if not Path(source).exists():
                raise FileNotFoundError(f"camera {self.cfg.id!r}: no such video {source}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"camera {self.cfg.id!r}: cannot open source {source!r}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.size[0]
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.size[1]
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.size = (w, h)
        if fps and 1 < fps < 240:
            self.fps = float(fps)
        self.cap = cap

    def _blank(self) -> np.ndarray:
        w, h = self.size
        frame = np.full((h, w, 3), (38, 36, 34), dtype=np.uint8)
        step = 40
        for x in range(0, w, step):
            cv2.line(frame, (x, 0), (x, h), (46, 44, 42), 1)
        for y in range(0, h, step):
            cv2.line(frame, (0, y), (w, y), (46, 44, 42), 1)
        return frame

    def read(self) -> np.ndarray | None:
        """Next frame, or None when this camera is done."""
        if self.n_scripted is not None and self.frame_idx >= self.n_scripted and self.cap is None:
            return None
        if self.cap is None:
            return self._blank()
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def rewind(self) -> None:
        self.frame_idx = 0
        self.finished = False
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.occupancy = ZoneOccupancy(
            camera=self.cfg.id,
            zoneset=self.cfg.zones,
            coverage_threshold=self.app.coverage_threshold,
            confirm_frames=self.app.confirm_frames,
            exit_frames=self.app.exit_frames,
            lost_frames=self.app.lost_frames,
        )

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


@dataclass
class StepResult:
    frames: list[np.ndarray]
    frame_idx: int
    events: list[ZoneEvent] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    finished: bool = False
    fps: float = 0.0


class Pipeline:
    def __init__(
        self,
        cfg: AppConfig,
        backend: str = "auto",
        loop: bool = False,
        alert_dir: Path | None = None,
    ):
        self.cfg = cfg
        self.loop = loop
        self.alert_dir = alert_dir
        self.catalog = Catalog.load(cfg.catalog)
        self.cameras = [
            CameraRuntime(cam, cfg, self.catalog, self._backend_for(cam, backend))
            for cam in cfg.cameras
        ]
        self.backend_label = "+".join(sorted({c.backend for c in self.cameras}))
        self.fps_nominal = cfg.fps or max(c.fps for c in self.cameras)
        self.cart = Cart(self.catalog)
        self.gate = ScanGate(self.catalog, fps=self.fps_nominal)
        self.frame_idx = 0
        self.started = time.perf_counter()
        self._tick = time.perf_counter()
        self._measured_fps = 0.0
        self._banners: dict[str, tuple[str, int]] = {}  # camera id -> (text, last frame)

    @staticmethod
    def _backend_for(cam: CameraConfig, backend: str) -> str:
        if backend == "auto":
            return "scripted" if cam.detections is not None else "yolo"
        return backend

    # -- main loop -----------------------------------------------------------

    def step(self) -> StepResult:
        raw: list[tuple[CameraRuntime, np.ndarray | None]] = []
        for cam in self.cameras:
            frame = None if cam.finished else cam.read()
            if frame is None:
                cam.finished = True
            else:
                cam.last_frame = frame
            raw.append((cam, frame))

        if all(cam.finished for cam in self.cameras):
            if self.loop:
                self.reset()
                return self.step()
            frames = [
                self._render(cam, cam.last_frame, [], ended=True)
                for cam in self.cameras
                if cam.last_frame is not None
            ]
            return StepResult(frames, self.frame_idx, finished=True, fps=self._measured_fps)

        events: list[ZoneEvent] = []
        alerts: list[Alert] = []
        per_camera: list[tuple[CameraRuntime, np.ndarray, list[Detection]]] = []

        for cam, frame in raw:
            if frame is None:
                if cam.last_frame is not None:
                    per_camera.append((cam, cam.last_frame.copy(), []))
                continue
            dets = cam.detector(frame, cam.frame_idx)
            cam.detections = dets
            cam.cfg.zones.bind(frame.shape[1], frame.shape[0])
            cam_events = cam.occupancy.update(dets, self.frame_idx)
            for event in cam_events:
                self.cart.handle(event)
                alert = self.gate.handle(event)
                if alert is not None:
                    alerts.append(alert)
            events.extend(cam_events)
            cam.frame_idx += 1
            per_camera.append((cam, frame, dets))

        for alert in alerts:
            self._banners[alert.camera] = (
                f"UNSCANNED ITEM IN BAG: {alert.display}",
                self.frame_idx + int(self.fps_nominal * 2.5),
            )

        frames = [self._render(cam, frame, dets) for cam, frame, dets in per_camera]

        if alerts and self.alert_dir is not None:
            self._save_snapshots(alerts, per_camera, frames)

        now = time.perf_counter()
        dt = now - self._tick
        self._tick = now
        if dt > 0:
            self._measured_fps = 0.85 * self._measured_fps + 0.15 * (1.0 / dt)

        self.frame_idx += 1
        return StepResult(
            frames=frames,
            frame_idx=self.frame_idx,
            events=events,
            alerts=alerts,
            finished=False,
            fps=self._measured_fps,
        )

    # -- rendering -----------------------------------------------------------

    def _render(
        self,
        cam: CameraRuntime,
        frame: np.ndarray,
        dets: list[Detection],
        ended: bool = False,
    ) -> np.ndarray:
        canvas = draw_zones(frame.copy(), cam.cfg.zones)
        draw_detections(canvas, self._rows(cam, dets))
        title = cam.cfg.label or cam.cfg.id
        roles = ",".join(sorted({z.role for z in cam.cfg.zones}))
        subtitle = f"{cam.backend}  |  {roles}" + ("  |  ended" if ended else "")
        draw_camera_header(canvas, title, subtitle)
        banner = self._banners.get(cam.cfg.id)
        if banner is not None and self.frame_idx <= banner[1]:
            draw_alert_banner(canvas, banner[0])
        return canvas

    def _rows(self, cam: CameraRuntime, dets: list[Detection]) -> list[dict]:
        rows = []
        for det in dets:
            key = (cam.cfg.id, det.track_id)
            if det.track_id is None:
                state = "untracked"
            elif key in self.gate.flagged_tracks:
                state = "flagged"
            elif key in self.gate.accounted_tracks:
                state = "accounted"
            else:
                state = "tracking"
            product = self.catalog.get(det.cls_name)
            name = product.display if product else det.cls_name
            tag = f"#{det.track_id} " if det.track_id is not None else ""
            text = f"{tag}{name} {det.conf:.2f}"
            if key in self.cart.lines:
                text += f"  {self.catalog.money(self.cart.lines[key].price)}"
            elif state == "flagged":
                text += "  UNSCANNED"
            rows.append({"box": det.box, "text": text, "state": state})
        return rows

    def _save_snapshots(
        self,
        alerts: list[Alert],
        per_camera: list[tuple[CameraRuntime, np.ndarray, list[Detection]]],
        frames: list[np.ndarray],
    ) -> None:
        assert self.alert_dir is not None
        self.alert_dir.mkdir(parents=True, exist_ok=True)
        by_cam = {cam.cfg.id: frames[i] for i, (cam, _, _) in enumerate(per_camera)}
        for alert in alerts:
            frame = by_cam.get(alert.camera)
            if frame is None:
                continue
            name = f"alert_f{alert.frame_idx:05d}_{alert.camera}_{alert.cls_name}.jpg"
            path = self.alert_dir / name
            cv2.imwrite(str(path), frame)
            alert.snapshot = str(path)

    # -- session -------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "frame": self.frame_idx,
            "fps": self._measured_fps,
            "backend": self.backend_label,
        }

    def reset(self) -> None:
        for cam in self.cameras:
            cam.rewind()
        self.cart = Cart(self.catalog)
        self.gate = ScanGate(self.catalog, fps=self.fps_nominal)
        self.frame_idx = 0
        self._banners.clear()

    def report(self) -> dict:
        return {
            "config": str(self.cfg.path),
            "backend": self.backend_label,
            "frames": self.frame_idx,
            "elapsed_s": round(time.perf_counter() - self.started, 2),
            "cameras": [
                {"id": c.cfg.id, "source": c.cfg.source, "zones": c.cfg.zones.to_list()}
                for c in self.cameras
            ],
            "receipt": self.cart.receipt(),
            "loss_prevention": self.gate.summary(),
            "cart_log": self.cart.log,
            "gate_log": self.gate.log,
        }

    def close(self) -> None:
        for cam in self.cameras:
            cam.close()
