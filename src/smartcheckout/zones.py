"""Polygonal zones and how much of a detection box falls inside one.

Zone polygons are stored in normalized [0, 1] image coordinates so a config
written against one camera resolution still works at another. Coverage is
measured by rasterizing each polygon once per frame size and reading box sums
off an integral image, which keeps concave zones exact and the per-box cost
constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import cv2
import numpy as np

#: A zone's role is what the pipeline does with the items inside it.
ROLES = ("bin", "basket", "scan", "bag")

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass
class Zone:
    name: str
    role: str
    polygon: list[tuple[float, float]]  # normalized (x, y) pairs
    _mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    _integral: np.ndarray | None = field(default=None, repr=False, compare=False)
    _size: tuple[int, int] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"zone {self.name!r}: role must be one of {ROLES}, got {self.role!r}")
        if len(self.polygon) < 3:
            raise ValueError(f"zone {self.name!r}: a polygon needs at least 3 points")
        self.polygon = [(float(x), float(y)) for x, y in self.polygon]

    @classmethod
    def from_dict(cls, data: dict) -> "Zone":
        return cls(
            name=data["name"],
            role=data["role"].lower(),
            polygon=[tuple(pt) for pt in data["polygon"]],
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "polygon": [[round(x, 5), round(y, 5)] for x, y in self.polygon],
        }

    def pixel_polygon(self, width: int, height: int) -> np.ndarray:
        pts = np.array(self.polygon, dtype=np.float64) * np.array([width, height])
        return np.round(pts).astype(np.int32)

    def bind(self, width: int, height: int) -> None:
        """Rasterize this zone for a given frame size (idempotent)."""
        if self._size == (width, height):
            return
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [self.pixel_polygon(width, height)], 1)
        self._mask = mask
        self._integral = cv2.integral(mask, sdepth=cv2.CV_32S)
        self._size = (width, height)

    def coverage(self, box: Box) -> float:
        """Fraction of ``box``'s area that lies inside the zone, in [0, 1]."""
        if self._integral is None or self._size is None:
            raise RuntimeError(f"zone {self.name!r} is not bound to a frame size")
        width, height = self._size
        x1 = int(np.clip(np.floor(box[0]), 0, width))
        y1 = int(np.clip(np.floor(box[1]), 0, height))
        x2 = int(np.clip(np.ceil(box[2]), 0, width))
        y2 = int(np.clip(np.ceil(box[3]), 0, height))
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            return 0.0
        ii = self._integral
        inside = int(ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1])
        return inside / area


class ZoneSet:
    """The zones belonging to one camera."""

    def __init__(self, zones: Iterable[Zone]):
        self.zones: list[Zone] = list(zones)
        names = [z.name for z in self.zones]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"duplicate zone names: {sorted(duplicates)}")

    @classmethod
    def from_list(cls, rows: Sequence[dict]) -> "ZoneSet":
        return cls(Zone.from_dict(row) for row in rows)

    def to_list(self) -> list[dict]:
        return [z.to_dict() for z in self.zones]

    def bind(self, width: int, height: int) -> None:
        for zone in self.zones:
            zone.bind(width, height)

    def by_role(self, role: str) -> list[Zone]:
        return [z for z in self.zones if z.role == role]

    def coverages(self, box: Box, threshold: float) -> dict[str, float]:
        """Zones whose coverage of ``box`` meets ``threshold``, name -> coverage."""
        hits = {}
        for zone in self.zones:
            cov = zone.coverage(box)
            if cov >= threshold:
                hits[zone.name] = cov
        return hits

    def get(self, name: str) -> Zone | None:
        for zone in self.zones:
            if zone.name == name:
                return zone
        return None

    def __iter__(self):
        return iter(self.zones)

    def __len__(self) -> int:
        return len(self.zones)
