"""Detection + tracking backends.

Every backend returns the same ``Detection`` records, so the checkout and
loss-prevention logic never knows whether the boxes came from a neural network
or from a scripted demo file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


@dataclass
class Detection:
    cls_name: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


class Detector(Protocol):
    def __call__(self, frame: np.ndarray, frame_idx: int) -> list[Detection]: ...


class YoloDetector:
    """Ultralytics detection + ByteTrack tracking, restricted to catalog classes.

    One instance owns one model, because the tracker state that ``persist=True``
    carries between calls belongs to a single camera.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        allowed_classes: Sequence[str] | None = None,
        min_conf: float = 0.35,
        imgsz: int = 640,
        device: str | None = None,
        tracker: str = "bytetrack.yaml",
    ):
        from ultralytics import YOLO  # imported lazily: heavy, and optional for demos

        self.model = YOLO(model_name)
        self.min_conf = min_conf
        self.imgsz = imgsz
        self.device = device
        self.tracker = tracker
        self.names: dict[int, str] = dict(self.model.names)
        self.class_ids: list[int] | None = None
        if allowed_classes:
            wanted = set(allowed_classes)
            self.class_ids = sorted(i for i, n in self.names.items() if n in wanted)
            missing = wanted - {self.names[i] for i in self.class_ids}
            if missing:
                raise ValueError(
                    "catalog classes the model cannot detect: " + ", ".join(sorted(missing))
                )

    def __call__(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            conf=self.min_conf,
            imgsz=self.imgsz,
            classes=self.class_ids,
            tracker=self.tracker,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or boxes.shape[0] == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        out = []
        for i in range(len(xyxy)):
            out.append(
                Detection(
                    cls_name=self.names.get(int(clss[i]), str(clss[i])),
                    conf=float(confs[i]),
                    box=tuple(float(v) for v in xyxy[i]),
                    track_id=int(ids[i]) if ids is not None else None,
                )
            )
        return out


class ScriptedDetector:
    """Replays detections recorded in a JSON sidecar.

    Lets the full pipeline -- zones, cart, alerts, overlay -- run with no model
    weights and no camera, which is what ``smartcheckout demo`` uses and what
    the tests exercise.
    """

    def __init__(self, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.width = int(data.get("width", 0))
        self.height = int(data.get("height", 0))
        self.fps = float(data.get("fps", 20.0))
        self.frames: dict[int, list[Detection]] = {}
        for row in data["frames"]:
            self.frames[int(row["frame"])] = [
                Detection(
                    cls_name=d["cls_name"],
                    conf=float(d.get("conf", 1.0)),
                    box=tuple(float(v) for v in d["box"]),
                    track_id=None if d.get("track_id") is None else int(d["track_id"]),
                )
                for d in row.get("detections", [])
            ]
        self.n_frames = (max(self.frames) + 1) if self.frames else 0

    def __call__(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        return list(self.frames.get(frame_idx, []))
