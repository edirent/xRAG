#!/usr/bin/env python
"""Train the O1-only C1 zero-gated residual architecture probe."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.train_set_fuser import main


if __name__ == "__main__":
    if "--full" in sys.argv:
        sys.argv.remove("--full")
        from scripts.packet_xrag.train_full_composition import main as full_main
        full_main(sys.argv[1:])
    else:
        main(["--branch", "C1", *sys.argv[1:]])
