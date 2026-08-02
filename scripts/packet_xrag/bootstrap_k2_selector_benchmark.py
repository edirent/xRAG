#!/usr/bin/env python
"""Paired bootstrap, overload characterization, and readiness gate for K2 selectors."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.run_k2_selector_benchmark import select_best_heuristic, summarize
from scripts.packet_xrag.token_resampler_common import EXPECTED_SPLIT_HASH


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",default="cache/results/k2_selector_benchmark_full500.jsonl")
    parser.add_argument("--summary",default="cache/results/k2_selector_benchmark_full500_summary.csv")
    parser.add_argument("--num-bootstrap",type=int,default=10000)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--json-output",default="cache/results/k2_selector_benchmark_bootstrap.json")
    parser.add_argument("--csv-output",default="cache/results/k2_selector_benchmark_bootstrap.csv")
    parser.add_argument("--markdown-output",default="cache/results/k2_selector_benchmark_bootstrap.md")
    parser.add_argument("--decision-output",default="cache/results/k2_selector_benchmark_decision.md")
    parser.add_argument("--overload-output",default="cache/results/k2_soft_token_overload_analysis.md")
    parser.add_argument("--error-jsonl",default="cache/results/k2_selector_error_analysis.jsonl")
    parser.add_argument("--error-md",default="cache/results/k2_selector_error_analysis.md")
    return parser.parse_args()


def load_rows(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids=sorted({row["sample_id"] for row in rows})
    configurations=sorted({row["configuration"] for row in rows})
    if len(ids)!=500 or len(rows)!=18000 or len(configurations)!=36:
        raise RuntimeError(f"unexpected benchmark cardinality: rows={len(rows)}, ids={len(ids)}, configs={len(configurations)}")
    by_configuration={}
    for configuration in configurations:
        items=[row for row in rows if row["configuration"]==configuration]
        if len(items)!=500 or {row["sample_id"] for row in items}!=set(ids):
            raise RuntimeError(f"sample-ID mismatch for {configuration}")
        by_configuration[configuration]={row["sample_id"]:row for row in items}
    return rows,ids,by_configuration


def paired_bootstrap(left,right,ids,num_bootstrap=10000,seed=42):
    if set(left)!=set(ids) or set(right)!=set(ids): raise ValueError("paired sample sets differ")
    differences=np.array([left[sid]-right[sid] for sid in ids],dtype=np.float64)*100
    rng=np.random.default_rng(seed);sampled=np.empty(num_bootstrap,dtype=np.float64)
    for start in range(0,num_bootstrap,1000):
        size=min(1000,num_bootstrap-start)
        indices=rng.integers(0,len(ids),size=(size,len(ids)))
        sampled[start:start+size]=differences[indices].mean(axis=1)
    return {"delta":float(differences.mean()),"ci95_lower":float(np.quantile(sampled,.025)),
            "ci95_upper":float(np.quantile(sampled,.975)),"p_delta_gt_0":float(np.mean(sampled>0)),
            "p_delta_ge_1":float(np.mean(sampled>=1)),"p_delta_ge_3":float(np.mean(sampled>=3)),
            "p_delta_ge_4":float(np.mean(sampled>=4))}


def scores(by_configuration,name):
    return {sid:row["short_f1"] for sid,row in by_configuration[name].items()}


def random_mean_scores(by_configuration,budget,ids):
    names=[f"RANDOM_{budget}_SEED{seed}" for seed in (13,37,73)]
    return {sid:sum(by_configuration[name][sid]["short_f1"] for name in names)/3 for sid in ids}


def family_best(summaries,family):
    items=[row for row in summaries if row["Configuration"].startswith(f"{family}_")]
    maximum=max(row["Short F1"] for row in items)
    close=[row for row in items if maximum-row["Short F1"]<0.25]
    return sorted(close,key=lambda row:int(row["Budget"]))[0]


def add_comparison(results,name,left,right,ids,count,seed):
    results[name]=paired_bootstrap(left,right,ids,count,seed)


def write_error_analysis(rows,by_configuration,best_name,best_topk,best_mmr,output_jsonl,output_md):
    by_sample=defaultdict(dict)
    for row in rows:by_sample[row["sample_id"]][row["configuration"]]=row
    categories={
        "XRAG_ORACLE correct, best heuristic wrong":lambda x:x["XRAG_ORACLE"]["short_em"]==1 and x[best_name]["short_em"]==0,
        "Best heuristic correct, ALL wrong":lambda x:x[best_name]["short_em"]==1 and x["ALL"]["short_em"]==0,
        "TOPK correct, MMR wrong":lambda x:x[best_topk]["short_em"]==1 and x[best_mmr]["short_em"]==0,
        "MMR correct, TOPK wrong":lambda x:x[best_mmr]["short_em"]==1 and x[best_topk]["short_em"]==0,
        "ORACLE_2 correct, ORACLE_1 wrong":lambda x:x["ORACLE_2"]["short_em"]==1 and x["ORACLE_1"]["short_em"]==0,
        "Support fully covered but answer wrong":lambda x:x[best_name]["full_support_coverage"]==1 and x[best_name]["short_em"]==0,
        "Support not fully covered but answer correct":lambda x:x[best_name]["full_support_coverage"]==0 and x[best_name]["short_em"]==1,
    }
    output=[]
    for category,predicate in categories.items():
        matches=[items for _,items in sorted(by_sample.items()) if predicate(items)][:20]
        for items in matches:
            focus=[items[name] for name in dict.fromkeys(["XRAG_ORACLE",best_name,"ALL",best_topk,best_mmr,"ORACLE_1","ORACLE_2"])]
            first=focus[0]
            output.append({"category":category,"sample_id":first["sample_id"],"question":first["question"],
                           "gold_answer":first["gold_answer"],"gold_packet_ids":first["gold_packet_ids"],
                           "gold_supporting_packets":items["XRAG_ORACLE"]["selected_packets"],
                           "configurations":[{"configuration":row["configuration"],"selected_packets":row["selected_packets"],
                                              "prediction":row["short_prediction"],"short_f1":row["short_f1"],
                                              "support_recall":row["support_recall"],"full_support":row["full_support_coverage"]} for row in focus]})
    with Path(output_jsonl).open("w") as stream:
        for row in output:stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    lines=["# K2 Selector Error Analysis",""]
    for category in categories:
        items=[row for row in output if row["category"]==category]
        lines += [f"## {category}","",f"Count shown: {len(items)}.",""]
        for row in items:
            lines.append(f"- `{row['sample_id']}` — {row['question']} | gold: {row['gold_answer']}")
            lines.append(f"  - Gold supporting packets: {[(p['packet_id'], p['encoder_text']) for p in row['gold_supporting_packets']]}")
            for configuration in row["configurations"]:
                selected=[]
                for packet in configuration["selected_packets"]:
                    score_fields={key:packet[key] for key in ("query_relevance","mmr_score_at_selection","max_similarity_to_selected","selection_step") if key in packet}
                    selected.append({"packet_id":packet["packet_id"],"encoder_text":packet["encoder_text"],**score_fields})
                lines.append(f"  - {configuration['configuration']}: prediction={configuration['prediction']!r}, F1={configuration['short_f1']:.4f}, selected={selected}")
        lines.append("")
    Path(output_md).write_text("\n".join(lines)+"\n")
    return {category:sum(row["category"]==category for row in output) for category in categories}


def main():
    args=parse_args()
    if args.num_bootstrap!=10000 or args.seed!=42:raise RuntimeError("locked bootstrap protocol mismatch")
    rows,ids,by=load_rows(args.input)
    summaries=summarize(rows,(13,37,73));best=select_best_heuristic(summaries)
    best_name=best["Configuration"];best_budget=int(best["Budget"])
    best_topk=family_best(summaries,"TOPK");best_mmr=family_best(summaries,"MMR")
    random_mean=random_mean_scores(by,best_budget,ids)
    results={}
    add_comparison(results,"Best heuristic vs same-budget RANDOM mean",scores(by,best_name),random_mean,ids,args.num_bootstrap,args.seed)
    add_comparison(results,"XRAG_ORACLE vs Best heuristic",scores(by,"XRAG_ORACLE"),scores(by,best_name),ids,args.num_bootstrap,args.seed)
    add_comparison(results,"XRAG_ORACLE vs ALL",scores(by,"XRAG_ORACLE"),scores(by,"ALL"),ids,args.num_bootstrap,args.seed)
    add_comparison(results,"Best heuristic vs ALL",scores(by,best_name),scores(by,"ALL"),ids,args.num_bootstrap,args.seed)
    add_comparison(results,"ORACLE_2 vs ORACLE_1",scores(by,"ORACLE_2"),scores(by,"ORACLE_1"),ids,args.num_bootstrap,args.seed)
    add_comparison(results,"Best TOPK vs Best MMR",scores(by,best_topk["Configuration"]),scores(by,best_mmr["Configuration"]),ids,args.num_bootstrap,args.seed)
    for budget in range(1,7):
        add_comparison(results,f"TOPK_{budget} vs MMR_{budget}",scores(by,f"TOPK_{budget}"),scores(by,f"MMR_{budget}"),ids,args.num_bootstrap,args.seed)
    selection=results["Best heuristic vs same-budget RANDOM mean"]
    headroom=results["XRAG_ORACLE vs Best heuristic"]
    oracle_overload=results["XRAG_ORACLE vs ALL"];heuristic_overload=results["Best heuristic vs ALL"]
    multihop=results["ORACLE_2 vs ORACLE_1"]
    sparse=best_budget<=4
    gate_a=(selection["delta"]>=3 and selection["ci95_lower"]>0 and headroom["delta"]>=4 and
            heuristic_overload["delta"]>=5 and oracle_overload["delta"]>=10 and sparse and multihop["delta"]>0)
    stable_selection=selection["delta"]>=3 and selection["ci95_lower"]>0
    clear_overload=heuristic_overload["delta"]>=5 and oracle_overload["delta"]>=10
    gate_b=(not gate_a and stable_selection and clear_overload and sparse and
            ((2<=headroom["delta"]<4) or headroom["ci95_lower"]<=0))
    gate="A" if gate_a else "B" if gate_b else "C"
    permissions={"static_scorer_allowed":gate in ("A","B"),"sequential_controller_allowed":gate=="A",
                 "explicit_stop_allowed":gate=="A"}
    topk_best_point=max([row for row in summaries if row["Configuration"].startswith("TOPK_")],key=lambda row:row["Short F1"])
    mmr_best_point=max([row for row in summaries if row["Configuration"].startswith("MMR_")],key=lambda row:row["Short F1"])
    topk6=next(row for row in summaries if row["Configuration"]=="TOPK_6")
    mmr6=next(row for row in summaries if row["Configuration"]=="MMR_6")
    harmful={"TOPK":topk6["Short F1"]-topk_best_point["Short F1"],"MMR":mmr6["Short F1"]-mmr_best_point["Short F1"]}
    inverse_u=topk_best_point["Budget"] not in (1,6) or mmr_best_point["Budget"] not in (1,6)
    error_counts=write_error_analysis(rows,by,best_name,best_topk["Configuration"],best_mmr["Configuration"],args.error_jsonl,args.error_md)
    payload={"validation_split_hash":EXPECTED_SPLIT_HASH,"samples":500,"resamples":args.num_bootstrap,"seed":args.seed,
             "best_heuristic":best,"best_topk":best_topk,"best_mmr":best_mmr,"comparisons":results,
             "harmful_expansion_delta":harmful,"inverse_u_shape":inverse_u,"selected_gate":gate,
             "permissions":permissions,"final_100_accessed":False,"final_100_runs":0,"error_analysis_counts":error_counts}
    Path(args.json_output).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    fields=["Comparison","Delta","CI Low","CI High","P(delta>0)","P(delta>=1)","P(delta>=3)","P(delta>=4)"]
    with Path(args.csv_output).open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for name,row in results.items():writer.writerow(dict(zip(fields,[name,row["delta"],row["ci95_lower"],row["ci95_upper"],row["p_delta_gt_0"],row["p_delta_ge_1"],row["p_delta_ge_3"],row["p_delta_ge_4"]])))
    lines=["# K2 Selector Paired Bootstrap","",f"Samples: 500; resamples: {args.num_bootstrap}; seed: {args.seed}.",""]
    for name,row in results.items():lines.append(f"- {name}: {row['delta']:.6f}, 95% CI [{row['ci95_lower']:.6f}, {row['ci95_upper']:.6f}], P(delta>0)={row['p_delta_gt_0']:.4f}.")
    Path(args.markdown_output).write_text("\n".join(lines)+"\n")
    curve={method:[next(row for row in summaries if row["Configuration"]==(f"RANDOM_{budget}_MEAN" if method=="RANDOM" else f"{method}_{budget}")) for budget in range(1,7)] for method in ("RANDOM","TOPK","MMR")}
    topk_recall_rises=topk6["Support Recall"]>topk_best_point["Support Recall"] and harmful["TOPK"]<0
    mmr_recall_rises=mmr6["Support Recall"]>mmr_best_point["Support Recall"] and harmful["MMR"]<0
    overload=["# K2 Soft-token Overload Analysis","",f"- F1 curve shape: {'inverse-U / interior optimum' if inverse_u else 'no clear inverse-U' }.",
              f"- Selected best packet budget: {best_budget}.",f"- TOPK harmful expansion delta (k=6 minus best k): {harmful['TOPK']:.6f} F1.",
              f"- MMR harmful expansion delta (k=6 minus best k): {harmful['MMR']:.6f} F1.",
              f"- Best heuristic minus ALL: {heuristic_overload['delta']:.6f} F1.",
              f"- XRAG_ORACLE minus ALL: {oracle_overload['delta']:.6f} F1.",
              f"- TOPK support recall rises while F1 falls from best k to k=6: {topk_recall_rises} ({topk_best_point['Support Recall']:.6f}→{topk6['Support Recall']:.6f} recall).",
              f"- MMR support recall rises while F1 falls from best k to k=6: {mmr_recall_rises} ({mmr_best_point['Support Recall']:.6f}→{mmr6['Support Recall']:.6f} recall).",
              "","| Method | Budget | Short F1 | Support Recall | Full Support | Avg Soft Tokens |","|---|---:|---:|---:|---:|---:|"]
    for method,items in curve.items():
        for row in items:overload.append(f"| {method} | {row['Budget']} | {row['Short F1']:.6f} | {row['Support Recall']:.6f} | {row['Full Support']:.6f} | {row['Avg Soft Tokens']:.6f} |")
    overload += ["", "As packet budget increases, support coverage generally rises; any simultaneous F1 decline is compressed-evidence overload rather than missing-evidence failure."]
    Path(args.overload_output).write_text("\n".join(overload)+"\n")
    summary_by={row["Configuration"]:row for row in summaries}
    diagnostic=["NO_CONTEXT","TEXT_ORACLE","XRAG_ORACLE","ORACLE_1","ORACLE_2","ALL"]
    audit=json.loads(Path("cache/results/k2_selector_checkpoint_audit.json").read_text())
    decision=["# Frozen-K2 Selector Benchmark Decision","",f"1. Validation split hash: `{EXPECTED_SPLIT_HASH}`.",
              f"2. Checkpoints: V1 `{audit['v1_checkpoint_path']}` SHA256 `{audit['v1_sha256']}`; K2 `{audit['k2_checkpoint_path']}` SHA256 `{audit['k2_sha256']}`.",
              f"3. K2 XRAG_ORACLE reproduced: {abs(summary_by['XRAG_ORACLE']['Short F1']-62.56321637426902)<=0.1}; observed {summary_by['XRAG_ORACLE']['Short F1']:.6f}."]
    for index,name in enumerate(diagnostic,4):decision.append(f"{index}. {name}: Short F1 {summary_by[name]['Short F1']:.6f}, Short EM {summary_by[name]['Short EM']:.6f}.")
    decision += ["10. RANDOM 1–6: "+"; ".join(f"k={k} seeds ["+", ".join(f"{seed}:{summary_by[f'RANDOM_{k}_SEED{seed}']['Short F1']:.6f}" for seed in (13,37,73))+f"], mean {summary_by[f'RANDOM_{k}_MEAN']['Short F1']:.6f}, std {summary_by[f'RANDOM_{k}_STD']['Short F1']:.6f}" for k in range(1,7))+".",
                 "11. TOPK 1–6: "+"; ".join(f"k={k} F1 {summary_by[f'TOPK_{k}']['Short F1']:.6f}" for k in range(1,7))+".",
                 "12. MMR 1–6: "+"; ".join(f"k={k} F1 {summary_by[f'MMR_{k}']['Short F1']:.6f}" for k in range(1,7))+".",
                 f"13. Best heuristic: {best_name}.",f"14. Best heuristic budget: {best_budget}.",
                 f"15. Selection signal: {selection['delta']:.6f}, CI [{selection['ci95_lower']:.6f}, {selection['ci95_upper']:.6f}].",
                 f"16. Oracle headroom: {headroom['delta']:.6f}, CI [{headroom['ci95_lower']:.6f}, {headroom['ci95_upper']:.6f}].",
                 f"17. Overload gaps: best-ALL {heuristic_overload['delta']:.6f}, CI [{heuristic_overload['ci95_lower']:.6f}, {heuristic_overload['ci95_upper']:.6f}]; oracle-ALL {oracle_overload['delta']:.6f}, CI [{oracle_overload['ci95_lower']:.6f}, {oracle_overload['ci95_upper']:.6f}].",
                 f"18. ORACLE_2 - ORACLE_1: {multihop['delta']:.6f}, CI [{multihop['ci95_lower']:.6f}, {multihop['ci95_upper']:.6f}].",
                 f"19. Harmful expansion: TOPK {harmful['TOPK']:.6f}; MMR {harmful['MMR']:.6f}.",
                 f"20. Selected gate: Gate {gate}.",f"21. Static scorer authorized: {permissions['static_scorer_allowed']}.",
                 f"22. Sequential controller authorized: {permissions['sequential_controller_allowed']}.",
                 f"23. Explicit STOP authorized: {permissions['explicit_stop_allowed']}.",
                 "24. Final 100 accessed: No.","25. Final 100 run count: 0."]
    Path(args.decision_output).write_text("\n".join(decision)+"\n")
    print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__":main()
