"""Feature 1: price whatever settles in the checkout bin.

A confirmed entry into a ``bin`` zone adds one line to the cart, keyed by the
track that carried it, and a confirmed exit takes that line back off -- so the
shopper can pull an item out and the total follows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .catalog import Catalog
from .zonestate import ZoneEvent


@dataclass
class CartLine:
    key: tuple[str, int]  # (camera id, track id)
    cls_name: str
    display: str
    sku: str
    price: float
    camera: str
    zone: str
    added_frame: int
    recognized: bool = True


@dataclass
class CartGroup:
    display: str
    sku: str
    unit_price: float
    qty: int
    recognized: bool

    @property
    def subtotal(self) -> float:
        return self.unit_price * self.qty


@dataclass
class Cart:
    catalog: Catalog
    lines: dict[tuple[str, int], CartLine] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    added: int = 0
    removed: int = 0

    def handle(self, event: ZoneEvent) -> None:
        if event.role != "bin":
            return
        key = (event.camera, event.track.track_id)
        if event.kind == "enter":
            self._add(key, event)
        elif event.kind == "exit":
            self._remove(key, event)

    def _add(self, key: tuple[str, int], event: ZoneEvent) -> None:
        if key in self.lines:
            return
        cls_name = event.track.cls_name
        product = self.catalog.get(cls_name)
        if product is None:
            line = CartLine(
                key=key,
                cls_name=cls_name,
                display=f"Unrecognized ({cls_name})",
                sku="--",
                price=0.0,
                camera=event.camera,
                zone=event.zone.name,
                added_frame=event.frame_idx,
                recognized=False,
            )
        else:
            line = CartLine(
                key=key,
                cls_name=cls_name,
                display=product.display,
                sku=product.sku,
                price=product.price,
                camera=event.camera,
                zone=event.zone.name,
                added_frame=event.frame_idx,
            )
        self.lines[key] = line
        self.added += 1
        self.log.append(
            f"[f{event.frame_idx:05d}] + {line.display} {self.catalog.money(line.price)}"
            f" (track {key[1]} in {event.zone.name})"
        )

    def _remove(self, key: tuple[str, int], event: ZoneEvent) -> None:
        line = self.lines.pop(key, None)
        if line is None:
            return
        self.removed += 1
        self.log.append(
            f"[f{event.frame_idx:05d}] - {line.display} {self.catalog.money(line.price)}"
            f" (track {key[1]} left {event.zone.name})"
        )

    @property
    def total(self) -> float:
        return sum(line.price for line in self.lines.values())

    @property
    def needs_attention(self) -> bool:
        return any(not line.recognized for line in self.lines.values())

    def groups(self) -> list[CartGroup]:
        """Cart lines collapsed into one row per SKU, newest SKU last."""
        order: list[str] = []
        groups: dict[str, CartGroup] = {}
        for line in sorted(self.lines.values(), key=lambda l: l.added_frame):
            gkey = line.sku if line.recognized else line.display
            if gkey not in groups:
                groups[gkey] = CartGroup(
                    display=line.display,
                    sku=line.sku,
                    unit_price=line.price,
                    qty=0,
                    recognized=line.recognized,
                )
                order.append(gkey)
            groups[gkey].qty += 1
        return [groups[k] for k in order]

    def receipt(self) -> dict:
        return {
            "currency": self.catalog.currency,
            "items": [
                {
                    "sku": g.sku,
                    "display": g.display,
                    "qty": g.qty,
                    "unit_price": round(g.unit_price, 2),
                    "subtotal": round(g.subtotal, 2),
                    "recognized": g.recognized,
                }
                for g in self.groups()
            ],
            "total": round(self.total, 2),
            "items_added": self.added,
            "items_removed": self.removed,
            "needs_attention": self.needs_attention,
        }
