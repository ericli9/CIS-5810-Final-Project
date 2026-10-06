"""Product catalog: maps detector class names to SKUs and prices."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Product:
    cls_name: str
    sku: str
    display: str
    price: float


class Catalog:
    """Lookup table from a detector class name to a priced product."""

    def __init__(self, products: dict[str, Product], currency: str = "$"):
        self._products = products
        self.currency = currency

    @classmethod
    def load(cls, path: str | Path) -> "Catalog":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        products = {}
        for row in data["products"]:
            p = Product(
                cls_name=row["class"],
                sku=row["sku"],
                display=row.get("display", row["class"].title()),
                price=float(row["price"]),
            )
            products[p.cls_name] = p
        return cls(products, currency=data.get("currency", "$"))

    def get(self, cls_name: str) -> Product | None:
        return self._products.get(cls_name)

    def known_classes(self) -> list[str]:
        return sorted(self._products)

    def money(self, amount: float) -> str:
        return f"{self.currency}{amount:,.2f}"

    def __contains__(self, cls_name: object) -> bool:
        return cls_name in self._products
