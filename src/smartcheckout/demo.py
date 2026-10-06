"""Builds the offline demo: two synthetic camera feeds plus ground-truth tracks.

The demo exists so the whole system -- zones, debouncing, cart, cross-camera
scan gate, overlay -- can be run and graded with no camera, no weights and no
network. The synthetic frames are deliberately simple; the detections come from
the script below rather than from a model, which is why they are replayed by
:class:`~smartcheckout.detect.ScriptedDetector` instead of being inferred.

The scenario is the one in the project brief, with the checkout bin acting as
the scanning field of view:

* ``bin`` camera -- a banana, a bottle and an orange are placed in the bin and
  priced; a cup is placed in and then taken back out, so its line comes off the
  cart and its scan credit is revoked.
* ``bag`` camera -- the banana and the bottle move from basket to bag and clear,
  because the bin camera already accounted for them. An apple moves into the bag
  having never appeared in the bin camera at all, which is the alert.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

FPS = 20
N_FRAMES = 220
SIZE = (640, 360)

PRODUCT_STYLE = {
    # class name -> (fill BGR, sprite size in px)
    "banana": ((70, 210, 235), (74, 44)),
    "bottle": ((175, 135, 85), (40, 92)),
    "apple": ((62, 62, 205), (50, 50)),
    "orange": ((45, 150, 245), (52, 52)),
    "cup": ((205, 205, 205), (46, 56)),
}

BIN_ZONES = [
    {"name": "checkout-bin", "role": "bin", "polygon": [[0.17, 0.22], [0.83, 0.22], [0.90, 0.93], [0.10, 0.93]]},
]
BAG_ZONES = [
    {"name": "basket", "role": "basket", "polygon": [[0.03, 0.30], [0.31, 0.30], [0.31, 0.90], [0.03, 0.90]]},
    {"name": "shopping-bag", "role": "bag", "polygon": [[0.69, 0.30], [0.97, 0.30], [0.97, 0.90], [0.69, 0.90]]},
]


@dataclass
class Actor:
    track_id: int
    cls_name: str
    #: (frame, normalized center x, normalized center y), in ascending frame order
    keys: list[tuple[int, float, float]] = field(default_factory=list)

    def center_at(self, frame: int) -> tuple[float, float] | None:
        if frame < self.keys[0][0] or frame > self.keys[-1][0]:
            return None
        for (f0, x0, y0), (f1, x1, y1) in zip(self.keys, self.keys[1:]):
            if f0 <= frame <= f1:
                t = 0.0 if f1 == f0 else (frame - f0) / (f1 - f0)
                # ease-in-out so motion looks like a hand, not a teleport
                t = t * t * (3 - 2 * t)
                return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
        f, x, y = self.keys[-1]
        return (x, y)


BIN_ACTORS = [
    Actor(1, "banana", [(0, 0.50, -0.15), (16, 0.34, 0.45), (N_FRAMES - 1, 0.34, 0.45)]),
    Actor(2, "bottle", [(26, 0.50, -0.18), (42, 0.63, 0.42), (N_FRAMES - 1, 0.63, 0.42)]),
    Actor(
        3,
        "cup",
        [
            (56, 0.50, -0.15),
            (72, 0.43, 0.74),
            (96, 0.43, 0.74),  # sits in the bin
            (120, 0.46, 0.05),  # lifted back out onto the counter
            (N_FRAMES - 1, 0.46, 0.05),
        ],
    ),
    Actor(4, "orange", [(150, 0.50, -0.15), (166, 0.73, 0.72), (N_FRAMES - 1, 0.73, 0.72)]),
]

BAG_ACTORS = [
    Actor(11, "banana", [(0, 0.16, 0.57), (34, 0.50, 0.57), (58, 0.84, 0.57), (N_FRAMES - 1, 0.84, 0.57)]),
    Actor(12, "bottle", [(0, 0.17, 0.77), (70, 0.17, 0.77), (102, 0.85, 0.77), (N_FRAMES - 1, 0.85, 0.77)]),
    # never seen by the bin camera -> unscanned item in the bag
    Actor(13, "apple", [(0, 0.16, 0.38), (130, 0.16, 0.38), (168, 0.84, 0.38), (N_FRAMES - 1, 0.84, 0.38)]),
]


def _background(kind: str) -> np.ndarray:
    w, h = SIZE
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    top = np.array((58, 54, 50), dtype=np.float64)
    bottom = np.array((30, 28, 26), dtype=np.float64)
    for y in range(h):
        frame[y, :] = top + (bottom - top) * (y / h)
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 3.5, (h, w, 1))
    frame = np.clip(frame.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    if kind == "bin":
        cv2.rectangle(frame, (int(0.08 * w), int(0.18 * h)), (int(0.92 * w), int(0.96 * h)), (44, 42, 40), -1)
        cv2.rectangle(frame, (int(0.08 * w), int(0.18 * h)), (int(0.92 * w), int(0.96 * h)), (70, 66, 62), 2)
        cv2.putText(frame, "checkout bin", (int(0.08 * w) + 8, int(0.18 * h) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 106, 100), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (int(0.02 * w), int(0.26 * h)), (int(0.33 * w), int(0.95 * h)), (46, 44, 42), -1)
        cv2.rectangle(frame, (int(0.67 * w), int(0.26 * h)), (int(0.99 * w), int(0.95 * h)), (46, 44, 42), -1)
        for x, text in ((0.02, "basket"), (0.67, "shopping bag")):
            cv2.putText(frame, text, (int(x * w) + 8, int(0.26 * h) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 106, 100), 1, cv2.LINE_AA)
    return frame


def _draw_product(frame: np.ndarray, cls_name: str, box: tuple[int, int, int, int]) -> None:
    fill, _ = PRODUCT_STYLE[cls_name]
    x1, y1, x2, y2 = box
    shade = tuple(int(c * 0.55) for c in fill)
    cv2.rectangle(frame, (x1 + 2, y1 + 3), (x2 + 3, y2 + 4), (18, 18, 18), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), shade, 2)
    cv2.putText(frame, cls_name[:3], (x1 + 4, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (25, 25, 25), 1, cv2.LINE_AA)


def _box_for(actor: Actor, frame_idx: int) -> tuple[float, float, float, float] | None:
    center = actor.center_at(frame_idx)
    if center is None:
        return None
    w, h = SIZE
    _, (sw, sh) = PRODUCT_STYLE[actor.cls_name]
    cx, cy = center[0] * w, center[1] * h
    box = (cx - sw / 2, cy - sh / 2, cx + sw / 2, cy + sh / 2)
    if box[2] < 0 or box[0] > w or box[3] < 0 or box[1] > h:
        return None
    return box


def _open_writer(path: Path) -> tuple[cv2.VideoWriter, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    if writer.isOpened():
        return writer, path
    fallback = path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(fallback), cv2.VideoWriter_fourcc(*"MJPG"), FPS, SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV cannot write video to {path} or {fallback}")
    return writer, fallback


def _render_feed(name: str, actors: list[Actor], out_dir: Path) -> tuple[Path, Path]:
    background = _background(name)
    writer, video_path = _open_writer(out_dir / f"{name}.mp4")
    rng = np.random.default_rng(11)
    records = []
    for frame_idx in range(N_FRAMES):
        frame = background.copy()
        dets = []
        for actor in actors:
            box = _box_for(actor, frame_idx)
            if box is None:
                continue
            ibox = tuple(int(round(v)) for v in box)
            _draw_product(frame, actor.cls_name, ibox)
            jitter = rng.normal(0, 1.2, 4)
            dets.append(
                {
                    "track_id": actor.track_id,
                    "cls_name": actor.cls_name,
                    "conf": round(float(rng.uniform(0.82, 0.96)), 3),
                    "box": [round(float(box[i] + jitter[i]), 1) for i in range(4)],
                }
            )
        writer.write(frame)
        records.append({"frame": frame_idx, "detections": dets})
    writer.release()

    det_path = out_dir / f"{name}.detections.json"
    det_path.write_text(
        json.dumps(
            {"width": SIZE[0], "height": SIZE[1], "fps": FPS, "frames": records}, indent=1
        )
        + "\n",
        encoding="utf-8",
    )
    return video_path, det_path


def build_demo(data_dir: Path, config_path: Path, catalog_rel: str = "catalog.json") -> Path:
    """Write the demo feeds into ``data_dir`` and a matching config to ``config_path``."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    bin_video, bin_dets = _render_feed("bin", BIN_ACTORS, data_dir)
    bag_video, bag_dets = _render_feed("bag", BAG_ACTORS, data_dir)

    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    def rel(p: Path) -> str:
        try:
            return p.resolve().relative_to(config_path.parent.resolve()).as_posix()
        except ValueError:
            return p.resolve().as_posix()

    config = {
        "catalog": catalog_rel,
        "detector": {"model": "yolo11n.pt", "min_conf": 0.35},
        "tuning": {
            "coverage_threshold": 0.30,
            "confirm_frames": 4,
            "exit_frames": 6,
            "lost_frames": 45,
            "fps": FPS,
        },
        "cameras": [
            {
                "id": "bin",
                "label": "CAM 1  checkout bin",
                "source": rel(bin_video),
                "detections": rel(bin_dets),
                "zones": BIN_ZONES,
            },
            {
                "id": "bag",
                "label": "CAM 2  basket -> shopping bag",
                "source": rel(bag_video),
                "detections": rel(bag_dets),
                "zones": BAG_ZONES,
            },
        ],
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path
