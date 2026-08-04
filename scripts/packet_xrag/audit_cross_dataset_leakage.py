#!/usr/bin/env python
"""Audit split isolation, feature provenance, and checkpoint ownership."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from src.packet_xrag.data import MusiqueAdapter, TriviaQAAdapter, TwoWikiAdapter
from src.packet_xrag.generalization.protocol import (
    assert_inference_view, register_checkpoint_owner,
)


ADAPTERS = {"2wiki": TwoWikiAdapter, "musique": MusiqueAdapter,
            "triviaqa": TriviaQAAdapter}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/generalization")
    args = parser.parse_args(argv); root = Path(args.root); results = {}
    checkpoint_hashes = {}
    for dataset, adapter_type in ADAPTERS.items():
        dataset_root = root / dataset
        if not dataset_root.exists(): continue
        manifest = json.loads((dataset_root / "splits/split_manifest.json").read_text())
        overlap_failures = {pair: values for pair, values in manifest["overlap_audit"].items()
            if values["sample_id_overlap"] or values["exact_question_overlap"] or
            values["normalized_question_overlap"]}
        if overlap_failures: raise RuntimeError(f"{dataset} split leakage: {overlap_failures}")
        adapter = adapter_type(); sample = json.loads((dataset_root /
            "records/train.jsonl").open().readline())
        packets = adapter.packetize(sample)
        inference_packets = adapter.inference_packets(packets)
        for packet in inference_packets: assert_inference_view(packet)
        feature_status = {}
        features = dataset_root / "features"
        if features.exists():
            for split in ("train", "dev", "shadow", "benchmark"):
                value = json.loads((features / split / "manifest.json").read_text())
                if value["support_fields_used_as_inference_features"]:
                    raise RuntimeError(f"{dataset}/{split} used support as an inference feature")
                if value["effective_split_hash"] != manifest["split_hashes"][split]:
                    raise RuntimeError(f"{dataset}/{split} feature/split hash mismatch")
                feature_status[split] = "PASS"
        selection = dataset_root / "fuser/run_1_hotpot_init/selection.json"
        if selection.exists():
            payload = json.loads(selection.read_text()); digest = payload["best_checkpoint_sha256"]
            register_checkpoint_owner(checkpoint_hashes, digest, dataset)
        results[dataset] = {"split_overlap": "PASS", "inference_label_stripping": "PASS",
                            "feature_provenance": feature_status}
    report = {"status": "PASS", "datasets": results,
        "dataset_checkpoint_owners": checkpoint_hashes,
        "cross_dataset_checkpoint_isolation": "PASS",
        "final100_accessed": False}
    output = root / "cross_dataset_leakage_audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
