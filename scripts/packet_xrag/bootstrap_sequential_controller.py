#!/usr/bin/env python
"""Paired bootstrap and Stage-2 selected-set conditioning gate."""

import argparse,json,sys
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:sys.path.insert(0,str(REPO_ROOT))
from scripts.packet_xrag.bootstrap_k2_selector_benchmark import paired_bootstrap


def rows(path,prefix):
    data=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    ids=sorted({x["sample_id"] for x in data});by={}
    for k in range(1,7):
        items={x["sample_id"]:x["short_f1"] for x in data if x["configuration"]==f"{prefix}_{k}"}
        if len(items)!=500:raise RuntimeError(f"{prefix}_{k} cardinality mismatch")
        by[f"{prefix}_{k}"]=items
    return ids,by


def main():
    p=argparse.ArgumentParser();p.add_argument("--seq-input",default="cache/controller/sequential_fixed_k/benchmark_predictions.jsonl")
    p.add_argument("--static-input",default="cache/controller/static/benchmark_predictions.jsonl")
    p.add_argument("--baseline-input",default="cache/results/k2_selector_benchmark_full500.jsonl")
    p.add_argument("--seq-config",default="cache/controller/sequential_fixed_k/best_short_f1/training_config.json")
    p.add_argument("--static-config",default="cache/controller/static/best_short_f1/training_config.json")
    p.add_argument("--output",default="cache/controller/sequential_fixed_k/sequential_bootstrap.json")
    p.add_argument("--decision",default="cache/controller/sequential_fixed_k/sequential_decision.md")
    a=p.parse_args();ids,seq=rows(a.seq_input,"SEQ");ids2,static=rows(a.static_input,"STATIC")
    if ids!=ids2:raise RuntimeError("SEQ/STATIC sample mismatch")
    baseline={}
    for line in Path(a.baseline_input).read_text().splitlines():
        x=json.loads(line)
        if x["configuration"].startswith("TOPK_"):baseline.setdefault(x["configuration"],{})[x["sample_id"]]=x["short_f1"]
    comparisons={}
    for k in range(1,7):
        comparisons[f"SEQ_{k} vs STATIC_{k}"]=paired_bootstrap(seq[f"SEQ_{k}"],static[f"STATIC_{k}"],ids,10000,42)
        comparisons[f"SEQ_{k} vs TOPK_{k}"]=paired_bootstrap(seq[f"SEQ_{k}"],baseline[f"TOPK_{k}"],ids,10000,42)
    sc=json.loads(Path(a.seq_config).read_text());st=json.loads(Path(a.static_config).read_text())
    sb=int(sc["selected_budget"]);tb=int(st["selected_budget"])
    gate=paired_bootstrap(seq[f"SEQ_{sb}"],static[f"STATIC_{tb}"],ids,10000,42)
    topk3=paired_bootstrap(seq[f"SEQ_{sb}"],baseline["TOPK_3"],ids,10000,42)
    passed=gate["delta"]>=1.5 and gate["ci95_lower"]>0
    payload={"samples":500,"resamples":10000,"seed":42,"selected_seq_budget":sb,"selected_static_budget":tb,
             "comparisons":comparisons,"gate_comparison":{"name":f"SEQ_{sb} vs STATIC_{tb}",**gate},
             "best_seq_vs_topk3":topk3,"sequential_gate_passed":passed,
             "next_action":"Stage 3" if passed else "Sequential state-usage audit",
             "benchmark_runs":1,"final_100_accessed":False,"final_100_runs":0}
    Path(a.output).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    Path(a.decision).write_text("# Sequential Fixed-k Decision\n\n"+
        f"- Gate: {'PASS' if passed else 'AUDIT REQUIRED'}\n- {payload['gate_comparison']['name']}: {gate['delta']:.6f}, 95% CI [{gate['ci95_lower']:.6f}, {gate['ci95_upper']:.6f}]\n"+
        f"- Next action: {payload['next_action']}\n- Final 100 accessed: no\n- Final 100 runs: 0\n")
    print(json.dumps(payload,indent=2),flush=True)

if __name__=="__main__":main()
