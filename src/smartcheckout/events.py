"""One structured record per thing the system decided.

The cart and the scan gate both append these. The CLI prints ``text``, the web
console renders the fields, and the session report serializes them -- so a
decision is described once, in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

#: ``kind`` values, in the order a shopper would cause them.
KINDS = ("add", "remove", "account", "revoke", "clear", "alert")


@dataclass
class LogEntry:
    frame: int
    kind: str
    text: str
    camera: str = ""
    display: str = ""

    def to_dict(self, fps: float = 0.0) -> dict:
        return {
            "frame": self.frame,
            "time_s": round(self.frame / fps, 2) if fps else 0.0,
            "kind": self.kind,
            "text": self.text,
            "camera": self.camera,
            "display": self.display,
        }
