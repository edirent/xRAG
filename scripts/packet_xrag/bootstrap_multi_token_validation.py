#!/usr/bin/env python
"""Paired bootstrap for K=1, K=2, and K=4 soft-token representations."""

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean


def percentile(values, probability):
    ordered = sorted(values); position = (len(ordered) - 1) * probability
    low = int(position); high = min(low + 1, len(ordered) - 1); fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def load_predictions(path, variant=None):
    records = {}
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if variant is not None and row.get("variant") != variant: continue
            sample_id = str(row["sample_id"])
            if sample_id in records: raise ValueError(f"duplicate sample {sample_id} in {path}:{line_number}")
            value = float(row["short_f1"])
            if not 0 <= value <= 1: raise ValueError("invalid Short F1")
            records[sample_id] = value
    if len(records) != 500: raise ValueError(f"expected 500 predictions in {path}, got {len(records)}")
    return records


def paired_bootstrap(representations, num_bootstrap=10_000, seed=42):
    ids = set(representations[1])
    if any(set(values) != ids for values in representations.values()): raise ValueError("paired sample IDs do not match")
    sample_ids = sorted(ids); arrays = {k: [v[sid] for sid in sample_ids] for k,v in representations.items()}
    comparisons = [(2,1),(4,1),(4,2)]; rng=random.Random(seed); draws={pair:[] for pair in comparisons}
    for _ in range(num_bootstrap):
        indices=[rng.randrange(500) for _ in range(500)]
        means={k:mean(values[i] for i in indices) for k,values in arrays.items()}
        for pair in comparisons: draws[pair].append(means[pair[0]]-means[pair[1]])
    result={"num_samples":500,"num_bootstrap":num_bootstrap,"seed":seed,"short_f1":{str(k):100*mean(v) for k,v in arrays.items()},"comparisons":{}}
    for upper,lower in comparisons:
        point=100*(mean(arrays[upper])-mean(arrays[lower])); values=draws[(upper,lower)]
        result["comparisons"][f"K{upper}_minus_K{lower}"]={"delta":point,
            "delta_ci95":[100*percentile(values,.025),100*percentile(values,.975)],
            "probability_positive":mean(value>0 for value in values),
            "probability_at_least_1":mean(100*value>=1 for value in values),
            "probability_at_least_3":mean(100*value>=3 for value in values)}
    return result


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k1",default="cache/results/packet_representation_ablation.jsonl")
    parser.add_argument("--k2",default="cache/projector/multi_token_k2/best_short_f1/validation_predictions.jsonl")
    parser.add_argument("--k4",default="cache/projector/multi_token_k4/best_short_f1/validation_predictions.jsonl")
    parser.add_argument("--num-bootstrap",type=int,default=10_000); parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--json-output",default="cache/results/multi_token_validation_bootstrap.json")
    parser.add_argument("--csv-output",default="cache/results/multi_token_validation_bootstrap.csv")
    parser.add_argument("--markdown-output",default="cache/results/multi_token_validation_bootstrap.md")
    return parser.parse_args()


def main():
    args=parse_args(); reps={1:load_predictions(args.k1,"V1_TITLE_SENTENCE"),2:load_predictions(args.k2),4:load_predictions(args.k4)}
    result=paired_bootstrap(reps,args.num_bootstrap,args.seed)
    Path(args.json_output).parent.mkdir(parents=True,exist_ok=True); Path(args.json_output).write_text(json.dumps(result,indent=2)+"\n")
    fields=["Comparison","Delta","CI Low","CI High","P(delta>0)","P(delta>=1)","P(delta>=3)"]
    with Path(args.csv_output).open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for name,row in result["comparisons"].items():writer.writerow(dict(zip(fields,[name,row["delta"],*row["delta_ci95"],row["probability_positive"],row["probability_at_least_1"],row["probability_at_least_3"]])))
    lines=["# Multi-token Validation Paired Bootstrap","",f"- Samples: 500; resamples: {args.num_bootstrap}; seed: {args.seed}.",""]
    for name,row in result["comparisons"].items():lines.append(f"- {name}: {row['delta']:.4f} F1 (95% CI {row['delta_ci95'][0]:.4f} to {row['delta_ci95'][1]:.4f}); P(delta>0)={row['probability_positive']:.4f}.")
    Path(args.markdown_output).write_text("\n".join(lines)+"\n");print(json.dumps(result,indent=2))


if __name__=="__main__":main()
