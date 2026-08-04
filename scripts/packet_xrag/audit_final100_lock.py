#!/usr/bin/env python
"""Verify the sealed Hotpot final-100 manifest without opening evaluation content."""

import argparse
import hashlib
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/generalization")
    parser.add_argument("--expect-runs", type=int, choices=(0, 1), default=0)
    args = parser.parse_args(argv); root = Path(args.root); final = root / "final100"
    lock = json.loads((final / "lock.json").read_text())
    ledger = json.loads((root / "experiment_ledger.json").read_text())
    global_audit = json.loads((root / "global_checkpoint_audit.json").read_text())
    sealed = json.loads((final / "sealed_ids.json").read_text())
    if lock["runs"] != args.expect_runs or lock["maximum_runs"] != 1:
        raise RuntimeError("final100 one-suite lock mismatch")
    if ledger["final100_suite_runs"] != args.expect_runs:
        raise RuntimeError("final100 ledger/lock mismatch")
    identifiers = sealed["ordered_sample_ids"]
    digest = hashlib.sha256("".join(identifiers).encode()).hexdigest()
    if digest != sealed["sha256"] or digest != global_audit["hotpot_final100_hash"]:
        raise RuntimeError("sealed final100 manifest changed")
    forbidden = [path.name for path in final.iterdir()
                 if path.name not in {"lock.json", "sealed_ids.json", "lock_audit.json"}]
    if args.expect_runs == 0 and forbidden:
        raise RuntimeError(f"final100 content exists before authorization: {forbidden}")
    report = {"status": "PASS", "sealed_sample_count": len(identifiers),
        "sealed_ids_sha256": digest, "suite_runs": args.expect_runs,
        "content_materialized": bool(forbidden), "lock_maximum_runs": 1}
    (final / "lock_audit.json").write_text(json.dumps(report, indent=2,
                                                       sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
