"""Feature 2: the scan gate.

An item is allowed in the bag only if it was accounted for first -- seen inside
a ``scan`` zone, or priced into the cart from a ``bin`` zone. Reaching a ``bag``
zone without that raises an alert.

Two independent checks run, because single-camera tracking is not reliable
enough on its own:

* **track identity** -- the same track id passed through the scan zone, which is
  exact when one camera sees both zones and the track survives;
* **class credits** -- a shared ledger counts accounted-for items per class, so
  an item scanned on *camera A* still clears when it shows up in the bag on
  *camera B*, and so an id switch mid-aisle does not create a false alarm.

An item whose class has no credit left in the ledger is the theft signal from
the project brief: it entered the bag field of view without ever passing
through the scan field of view.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .catalog import Catalog
from .zonestate import ZoneEvent

#: Roles that make an item "accounted for" and mint a credit in the ledger.
DEFAULT_CREDIT_ROLES = ("scan", "bin")


@dataclass
class Alert:
    frame_idx: int
    time_s: float
    camera: str
    zone: str
    track_id: int
    cls_name: str
    display: str
    reason: str
    snapshot: str | None = None

    def to_dict(self) -> dict:
        return {
            "frame": self.frame_idx,
            "time_s": round(self.time_s, 2),
            "camera": self.camera,
            "zone": self.zone,
            "track_id": self.track_id,
            "class": self.cls_name,
            "display": self.display,
            "reason": self.reason,
            "snapshot": self.snapshot,
        }


@dataclass
class ScanGate:
    catalog: Catalog
    fps: float = 20.0
    credit_roles: tuple[str, ...] = DEFAULT_CREDIT_ROLES
    credits: Counter = field(default_factory=Counter)
    accounted_tracks: set[tuple[str, int]] = field(default_factory=set)
    cleared: list[dict] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    flagged_tracks: set[tuple[str, int]] = field(default_factory=set)
    log: list[str] = field(default_factory=list)

    def handle(self, event: ZoneEvent) -> Alert | None:
        if event.kind == "exit":
            # Taking an item back out of the bin un-buys it, so its credit goes
            # away too -- otherwise a shopper could bin an item, lift it out and
            # walk it into the bag on the strength of a stale credit.
            if event.role == "bin":
                self._revoke(event)
            return None
        if event.kind != "enter":
            return None
        if event.role in self.credit_roles:
            self._account(event)
            return None
        if event.role == "bag":
            return self._gate(event)
        return None

    def _account(self, event: ZoneEvent) -> None:
        key = (event.camera, event.track.track_id)
        if key in self.accounted_tracks:
            return
        cls_name = event.track.cls_name
        self.accounted_tracks.add(key)
        self.credits[cls_name] += 1
        self.log.append(
            f"[f{event.frame_idx:05d}] scanned {self._display(cls_name)}"
            f" ({event.zone.name} on {event.camera})"
        )

    def _revoke(self, event: ZoneEvent) -> None:
        key = (event.camera, event.track.track_id)
        if key not in self.accounted_tracks:
            return
        cls_name = event.track.cls_name
        self.accounted_tracks.discard(key)
        self.credits[cls_name] = max(0, self.credits[cls_name] - 1)
        self.log.append(
            f"[f{event.frame_idx:05d}] credit revoked for {self._display(cls_name)}"
            f" (left {event.zone.name} on {event.camera})"
        )

    def _gate(self, event: ZoneEvent) -> Alert | None:
        key = (event.camera, event.track.track_id)
        cls_name = event.track.cls_name

        if key in self.accounted_tracks:
            self._clear(event, "same track passed the scan zone")
            return None
        if self.credits[cls_name] > 0:
            self.credits[cls_name] -= 1
            self.accounted_tracks.add(key)
            self._clear(event, "matched an accounted-for item of the same class")
            return None
        if key in self.flagged_tracks:
            return None

        alert = Alert(
            frame_idx=event.frame_idx,
            time_s=event.frame_idx / self.fps if self.fps else 0.0,
            camera=event.camera,
            zone=event.zone.name,
            track_id=event.track.track_id,
            cls_name=cls_name,
            display=self._display(cls_name),
            reason="entered the bag without passing a scan zone",
        )
        self.flagged_tracks.add(key)
        self.alerts.append(alert)
        self.log.append(
            f"[f{event.frame_idx:05d}] ALERT unscanned {alert.display}"
            f" in {event.zone.name} on {event.camera}"
        )
        return alert

    def _clear(self, event: ZoneEvent, why: str) -> None:
        self.cleared.append(
            {
                "frame": event.frame_idx,
                "camera": event.camera,
                "zone": event.zone.name,
                "track_id": event.track.track_id,
                "class": event.track.cls_name,
                "why": why,
            }
        )
        self.log.append(
            f"[f{event.frame_idx:05d}] ok {self._display(event.track.cls_name)}"
            f" bagged ({why})"
        )

    def _display(self, cls_name: str) -> str:
        product = self.catalog.get(cls_name)
        return product.display if product else cls_name.title()

    def summary(self) -> dict:
        return {
            "alerts": [a.to_dict() for a in self.alerts],
            "cleared": self.cleared,
            "unused_credits": {k: v for k, v in sorted(self.credits.items()) if v},
        }
