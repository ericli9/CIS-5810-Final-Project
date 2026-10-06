#!/usr/bin/env python
"""Entry point that works without installing the package.

    python run.py demo --run
    python run.py run --config configs/webcam-single.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from smartcheckout.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
