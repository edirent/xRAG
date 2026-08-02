#!/usr/bin/env python
"""Train the parameter-matched one-memory pooled resampler control."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.train_token_state_resampler import main


if __name__ == "__main__":
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", "cache/resampler/pooled_control"])
    main(memory_source="pooled")
