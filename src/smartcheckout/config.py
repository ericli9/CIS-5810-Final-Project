"""Loading and saving the scene configuration (cameras, zones, tunables)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .zones import ZoneSet


def resolve_path(value: str, base: Path) -> Path:
    """Resolve ``value`` against the config file's directory, then the cwd."""
    p = Path(value)
    if p.is_absolute():
        return p
    candidate = (base / p).resolve()
    if candidate.exists():
        return candidate
    return (Path.cwd() / p).resolve()


@dataclass
class CameraConfig:
    id: str
    source: str
    zones: ZoneSet
    detections: Path | None = None  # scripted sidecar, for the offline demo
    label: str = ""
    base: Path = field(default_factory=Path.cwd)  # directory the config was loaded from

    @property
    def capture_source(self) -> str | int:
        """A bare number means a webcam index, anything else is a path or URL."""
        source = str(self.source)
        if source.lstrip("-").isdigit():
            return int(source)
        if source.lower() in {"blank", "none", ""} or source.startswith(("http", "rtsp")):
            return source
        return str(resolve_path(source, self.base))

    def to_dict(self) -> dict:
        data: dict = {"id": self.id, "source": self.source, "zones": self.zones.to_list()}
        if self.label:
            data["label"] = self.label
        if self.detections is not None:
            data["detections"] = str(self.detections)
        return data


@dataclass
class AppConfig:
    cameras: list[CameraConfig]
    catalog: Path
    path: Path
    model: str = "yolo11n.pt"
    min_conf: float = 0.35
    imgsz: int = 640
    device: str | None = None
    tracker: str = "bytetrack.yaml"
    coverage_threshold: float = 0.30
    confirm_frames: int = 4
    exit_frames: int = 6
    lost_frames: int = 45
    fps: float | None = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        cfg_path = Path(path).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"config not found: {cfg_path}")
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        base = cfg_path.parent

        cameras = []
        for row in data.get("cameras", []):
            cameras.append(
                CameraConfig(
                    id=row["id"],
                    source=str(row["source"]),
                    zones=ZoneSet.from_list(row.get("zones", [])),
                    detections=(
                        resolve_path(row["detections"], base) if row.get("detections") else None
                    ),
                    label=row.get("label", ""),
                    base=base,
                )
            )
        if not cameras:
            raise ValueError(f"{cfg_path}: config defines no cameras")
        ids = [c.id for c in cameras]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{cfg_path}: duplicate camera ids {ids}")

        catalog_rel = data.get("catalog", "catalog.json")
        tun = data.get("tuning", {})
        det = data.get("detector", {})
        return cls(
            cameras=cameras,
            catalog=resolve_path(catalog_rel, base),
            path=cfg_path,
            model=det.get("model", "yolo11n.pt"),
            min_conf=float(det.get("min_conf", 0.35)),
            imgsz=int(det.get("imgsz", 640)),
            device=det.get("device"),
            tracker=det.get("tracker", "bytetrack.yaml"),
            coverage_threshold=float(tun.get("coverage_threshold", 0.30)),
            confirm_frames=int(tun.get("confirm_frames", 4)),
            exit_frames=int(tun.get("exit_frames", 6)),
            lost_frames=int(tun.get("lost_frames", 45)),
            fps=float(tun["fps"]) if tun.get("fps") else None,
            raw=data,
        )

    def save(self, path: str | Path | None = None) -> Path:
        out = Path(path) if path else self.path
        data = dict(self.raw)
        data["cameras"] = [c.to_dict() for c in self.cameras]
        data.setdefault("catalog", "catalog.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return out

    def camera(self, camera_id: str) -> CameraConfig:
        for cam in self.cameras:
            if cam.id == camera_id:
                return cam
        raise KeyError(f"no camera {camera_id!r} in {self.path} (have: {[c.id for c in self.cameras]})")
