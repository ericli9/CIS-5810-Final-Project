"""Interactive zone editor: click polygons onto a camera frame and save them."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig, CameraConfig
from .viz import FONT, ROLE_COLORS, put
from .zones import Zone

ROLE_KEYS = {ord("1"): "bin", ord("2"): "basket", ord("3"): "scan", ord("4"): "bag"}
HELP = [
    "click: add point      u: undo point      enter: close polygon",
    "1 bin   2 basket   3 scan   4 bag        d: delete last zone",
    "s: save to config     q: quit without saving",
]


def grab_frame(cam: CameraConfig) -> np.ndarray:
    source = cam.capture_source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source {source!r} for camera {cam.id!r}")
    frame = None
    for _ in range(10):  # let a webcam settle on a real exposure
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    if frame is None:
        raise RuntimeError(f"no frames read from source {source!r}")
    return frame


def edit_zones(cfg: AppConfig, camera_id: str, out_path: Path | None = None) -> bool:
    """Returns True if the config was saved."""
    cam = cfg.camera(camera_id)
    frame = grab_frame(cam)
    h, w = frame.shape[:2]
    zones: list[Zone] = list(cam.zones)
    points: list[tuple[int, int]] = []
    awaiting_role = False
    window = f"zone editor - {cam.id}"

    def on_mouse(event: int, x: int, y: int, flags: int, userdata) -> None:
        nonlocal points
        if awaiting_role:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    saved = False

    while True:
        canvas = frame.copy()
        overlay = canvas.copy()
        for zone in zones:
            color = ROLE_COLORS.get(zone.role, (200, 200, 200))
            poly = zone.pixel_polygon(w, h)
            cv2.fillPoly(overlay, [poly], color)
        canvas = cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0)
        for zone in zones:
            color = ROLE_COLORS.get(zone.role, (200, 200, 200))
            poly = zone.pixel_polygon(w, h)
            cv2.polylines(canvas, [poly], True, color, 2, cv2.LINE_AA)
            put(canvas, f"{zone.role}:{zone.name}", (int(poly[:, 0].min()) + 5,
                int(poly[:, 1].min()) + 18), 0.46, color, 1)

        if len(points) >= 2:
            cv2.polylines(canvas, [np.array(points, np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA)
        for pt in points:
            cv2.circle(canvas, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)

        band = canvas.shape[0] - 18 * len(HELP) - 26
        cv2.rectangle(canvas, (0, band), (canvas.shape[1], canvas.shape[0]), (24, 22, 20), -1)
        for i, line in enumerate(HELP):
            put(canvas, line, (12, band + 20 + i * 18), 0.44, (225, 225, 225))
        status = (
            "pick a role for the new polygon: 1 bin  2 basket  3 scan  4 bag"
            if awaiting_role
            else f"{len(zones)} zone(s), {len(points)} point(s) in progress"
        )
        put(canvas, status, (12, band + 20 + len(HELP) * 18),
            0.46, (90, 200, 255) if awaiting_role else (170, 170, 170))

        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue

        if awaiting_role:
            if key in ROLE_KEYS:
                role = ROLE_KEYS[key]
                name = f"{role}-{sum(1 for z in zones if z.role == role) + 1}"
                zones.append(
                    Zone(
                        name=name,
                        role=role,
                        polygon=[(x / w, y / h) for x, y in points],
                    )
                )
                points = []
                awaiting_role = False
            elif key in (27, ord("q")):
                awaiting_role = False
            continue

        if key in (ord("q"), 27):
            break
        if key == ord("u") and points:
            points.pop()
        elif key in (13, 10):  # enter
            if len(points) >= 3:
                awaiting_role = True
            else:
                print("a polygon needs at least 3 points")
        elif key == ord("d") and zones:
            dropped = zones.pop()
            print(f"deleted zone {dropped.name}")
        elif key == ord("s"):
            cam.zones.zones = zones
            target = cfg.save(out_path)
            print(f"saved {len(zones)} zone(s) for camera {cam.id!r} -> {target}")
            saved = True
            break

    cv2.destroyWindow(window)
    return saved
