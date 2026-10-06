"""Smart checkout: vision-based auto-checkout and scan-gate loss prevention.

Two features, one pipeline:

1. ``checkout`` -- items that settle inside a BIN zone are priced and added to a
   live cart, so the shopper never scans anything by hand.
2. ``loss prevention`` -- items that reach a BAG zone without ever having been
   seen inside a SCAN zone raise an alert. The SCAN zone may live in a different
   camera than the BAG zone; cameras share a credit ledger so the check works
   across fields of view.
"""

__version__ = "0.1.0"
