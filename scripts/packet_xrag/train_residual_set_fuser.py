#!/usr/bin/env python
"""Train the O1-only C1 zero-gated residual architecture probe."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.train_set_fuser import main


if __name__ == "__main__":
    main(["--branch", "C1", *sys.argv[1:]])
