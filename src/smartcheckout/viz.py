"""Overlay and dashboard rendering (OpenCV only, no GUI toolkit)."""

from __future__ import annotations

import cv2
import numpy as np

from .catalog import Catalog
from .checkout import Cart
from .lossprev import ScanGate
from .zones import ZoneSet

FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR
BG = (26, 24, 22)
PANEL = (34, 32, 29)
HAIRLINE = (62, 58, 54)
TEXT = (236, 238, 240)
MUTED = (150, 152, 156)
ACCENT = (210, 170, 90)

ROLE_COLORS = {
    "bin": (225, 170, 70),
    "scan": (120, 205, 120),
    "basket": (190, 150, 120),
    "bag": (90, 180, 235),
}
STATE_COLORS = {
    "accounted": (120, 205, 120),
    "flagged": (70, 70, 235),
    "tracking": (225, 225, 225),
    "untracked": (140, 140, 140),
}

PANEL_WIDTH = 400


def put(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.46,
    color: tuple[int, int, int] = TEXT,
    thick: int = 1,
) -> None:
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _label(img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.44, 1)
    y = max(y, th + 8)
    x = int(np.clip(x, 0, max(0, img.shape[1] - tw - 12)))  # keep labels on screen
    cv2.rectangle(img, (x, y - th - 7), (x + tw + 10, y + 3), color, -1)
    put(img, text, (x + 5, y - 2), 0.44, (20, 20, 20), 1)


def draw_zones(frame: np.ndarray, zoneset: ZoneSet) -> np.ndarray:
    h, w = frame.shape[:2]
    zoneset.bind(w, h)
    overlay = frame.copy()
    for zone in zoneset:
        color = ROLE_COLORS.get(zone.role, ACCENT)
        poly = zone.pixel_polygon(w, h)
        cv2.fillPoly(overlay, [poly], color)
    frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
    for zone in zoneset:
        color = ROLE_COLORS.get(zone.role, ACCENT)
        poly = zone.pixel_polygon(w, h)
        cv2.polylines(frame, [poly], True, color, 2, cv2.LINE_AA)
        x, y = poly[:, 0].min(), poly[:, 1].min()
        _label(frame, f"{zone.role.upper()} - {zone.name}", int(x) + 4, int(y) + 22, color)
    return frame


def draw_detections(frame: np.ndarray, rows: list[dict]) -> None:
    """``rows``: dicts with box, text, state (see :mod:`smartcheckout.pipeline`)."""
    for row in rows:
        x1, y1, x2, y2 = (int(round(v)) for v in row["box"])
        color = STATE_COLORS.get(row["state"], STATE_COLORS["tracking"])
        thick = 3 if row["state"] == "flagged" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
        _label(frame, row["text"], x1, y1 - 4, color)


def draw_camera_header(frame: np.ndarray, title: str, subtitle: str) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 30), BG, -1)
    put(frame, title, (12, 21), 0.56, TEXT, 1)
    (tw, _), _ = cv2.getTextSize(subtitle, FONT, 0.44, 1)
    put(frame, subtitle, (w - tw - 12, 21), 0.44, MUTED, 1)


def draw_alert_banner(frame: np.ndarray, text: str) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 42), (w, h), (50, 50, 190), -1)
    put(frame, text, (14, h - 15), 0.58, (255, 255, 255), 2)
    cv2.rectangle(frame, (2, 2), (w - 3, h - 3), (60, 60, 220), 4)


def _section(panel: np.ndarray, y: int, title: str) -> int:
    w = panel.shape[1]
    put(panel, title, (16, y), 0.5, ACCENT, 1)
    cv2.line(panel, (16, y + 8), (w - 16, y + 8), HAIRLINE, 1)
    return y + 28


def render_dashboard(
    height: int,
    catalog: Catalog,
    cart: Cart,
    gate: ScanGate,
    stats: dict,
    width: int = PANEL_WIDTH,
) -> np.ndarray:
    panel = np.full((height, width, 3), PANEL, dtype=np.uint8)
    cv2.line(panel, (0, 0), (0, height), HAIRLINE, 1)

    put(panel, "SMART CHECKOUT", (16, 32), 0.66, TEXT, 2)
    put(
        panel,
        f"frame {stats.get('frame', 0):>5}   {stats.get('fps', 0.0):.1f} fps"
        f"   {stats.get('backend', '')}",
        (16, 52),
        0.42,
        MUTED,
    )

    y = _section(panel, 82, "CART")
    groups = cart.groups()
    if not groups:
        put(panel, "bin is empty", (22, y), 0.44, MUTED)
        y += 22
    for group in groups[-10:]:
        color = TEXT if group.recognized else (90, 170, 245)
        name = group.display if len(group.display) <= 24 else group.display[:23] + "."
        put(panel, f"{group.qty}x  {name}", (22, y), 0.46, color)
        amount = catalog.money(group.subtotal)
        (tw, _), _ = cv2.getTextSize(amount, FONT, 0.46, 1)
        put(panel, amount, (width - tw - 20, y), 0.46, color)
        y += 23
    if len(groups) > 10:
        put(panel, f"... {len(groups) - 10} more", (22, y), 0.42, MUTED)
        y += 20

    y += 8
    cv2.line(panel, (16, y), (width - 16, y), HAIRLINE, 1)
    y += 30
    put(panel, "TOTAL", (20, y), 0.6, TEXT, 2)
    total = catalog.money(cart.total)
    (tw, _), _ = cv2.getTextSize(total, FONT, 0.78, 2)
    put(panel, total, (width - tw - 20, y + 2), 0.78, ACCENT, 2)
    y += 20
    if cart.needs_attention:
        y += 18
        put(panel, "unrecognized item - call attendant", (20, y), 0.42, (90, 170, 245))

    y = _section(panel, y + 42, "SCAN GATE")
    pending = sum(v for v in gate.credits.values() if v > 0)
    put(panel, f"accounted for : {len(gate.accounted_tracks)}", (22, y), 0.44, MUTED)
    y += 20
    put(panel, f"bagged / cleared : {len(gate.cleared)}", (22, y), 0.44, MUTED)
    y += 20
    put(panel, f"unused credits : {pending}", (22, y), 0.44, MUTED)

    y = _section(panel, y + 40, f"ALERTS ({len(gate.alerts)})")
    if not gate.alerts:
        put(panel, "none", (22, y), 0.44, (120, 205, 120))
    for alert in gate.alerts[-5:]:
        put(panel, f"t={alert.time_s:5.1f}s  {alert.display}", (22, y), 0.46, (90, 90, 245))
        y += 19
        put(panel, f"   unscanned -> {alert.zone} ({alert.camera})", (22, y), 0.4, MUTED)
        y += 24

    footer = "q quit   space pause   r reset"
    put(panel, footer, (16, height - 16), 0.42, MUTED)
    return panel


def tile(frames: list[np.ndarray], width: int) -> np.ndarray:
    """Stack camera views vertically at a common width."""
    scaled = []
    for frame in frames:
        h, w = frame.shape[:2]
        new_h = max(1, int(round(h * width / w)))
        scaled.append(cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA))
    gap = 6
    total_h = sum(f.shape[0] for f in scaled) + gap * (len(scaled) - 1)
    canvas = np.full((total_h, width, 3), BG, dtype=np.uint8)
    y = 0
    for frame in scaled:
        canvas[y : y + frame.shape[0]] = frame
        y += frame.shape[0] + gap
    return canvas


def compose(
    frames: list[np.ndarray],
    catalog: Catalog,
    cart: Cart,
    gate: ScanGate,
    stats: dict,
    view_width: int = 720,
) -> np.ndarray:
    views = tile(frames, view_width)
    height = max(views.shape[0], 640)
    canvas = np.full((height, view_width + PANEL_WIDTH, 3), BG, dtype=np.uint8)
    canvas[: views.shape[0], :view_width] = views
    canvas[:, view_width:] = render_dashboard(height, catalog, cart, gate, stats)
    return canvas
