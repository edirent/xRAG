#!/usr/bin/env python
"""Paired factorial bootstrap for K=2 capacity and frozen LoRA interaction."""

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean

CONFIGS=("K1_BASE","K1_LORA","K2_BASE","K2_LORA")


def percentile(values,p):
    ordered=sorted(values);position=(len(ordered)-1)*p;low=int(position);high=min(low+1,len(ordered)-1);fraction=position-low
    return ordered[low]*(1-fraction)+ordered[high]*fraction


def load(path,configuration=None,variant=None):
    result={}
    with Path(path).open() as stream:
        for number,line in enumerate(stream,1):
            row=json.loads(line)
            if configuration is not None and row.get("configuration")!=configuration:continue
            if variant is not None and row.get("variant")!=variant:continue
            sid=str(row["sample_id"])
            if sid in result:raise ValueError(f"duplicate {sid} in {path}:{number}")
            value=float(row["short_f1"])
            if not 0<=value<=1:raise ValueError("invalid F1")
            result[sid]=value
    if len(result)!=500:raise ValueError(f"expected 500 rows for {configuration or variant}, got {len(result)}")
    return result


def interval(draws):return [100*percentile(draws,.025),100*percentile(draws,.975)]


def analyze(data,num_bootstrap=10_000,seed=42):
    ids=set(data["K1_BASE"])
    if any(set(values)!=ids for values in data.values()):raise ValueError("paired sample IDs do not match")
    sample_ids=sorted(ids);arrays={name:[values[sid] for sid in sample_ids] for name,values in data.items()}
    pairs=[("K2_LORA","K1_BASE"),("K2_LORA","K2_BASE"),("K2_LORA","K1_LORA"),("K2_BASE","K1_BASE"),("K1_LORA","K1_BASE")]
    draws={f"{a}_minus_{b}":[] for a,b in pairs};interaction=[];rng=random.Random(seed)
    for _ in range(num_bootstrap):
        indices=[rng.randrange(500) for _ in range(500)];scores={name:mean(values[i] for i in indices) for name,values in arrays.items()}
        for a,b in pairs:draws[f"{a}_minus_{b}"].append(scores[a]-scores[b])
        interaction.append(scores["K2_LORA"]-scores["K2_BASE"]-scores["K1_LORA"]+scores["K1_BASE"])
    points={name:100*mean(values) for name,values in arrays.items()};comparisons={}
    for a,b in pairs:
        name=f"{a}_minus_{b}";values=draws[name];delta=points[a]-points[b]
        comparisons[name]={"delta":delta,"delta_ci95":interval(values),"probability_positive":mean(v>0 for v in values),
            "probability_at_least_1":mean(100*v>=1 for v in values),"probability_at_least_3":mean(100*v>=3 for v in values)}
    interaction_point=points["K2_LORA"]-points["K2_BASE"]-points["K1_LORA"]+points["K1_BASE"]
    return {"num_samples":500,"num_bootstrap":num_bootstrap,"seed":seed,"short_f1":points,"comparisons":comparisons,
        "interaction":{"point_estimate":interaction_point,"ci95":interval(interaction),"probability_positive":mean(v>0 for v in interaction),"probability_negative":mean(v<0 for v in interaction)}}


def parse_args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--k1-base",default="cache/results/packet_representation_ablation.jsonl")
    p.add_argument("--k1-lora",default="cache/results/k1_lora_validation_predictions_recovered.jsonl")
    p.add_argument("--k2-base",default="cache/projector/multi_token_k2/best_short_f1/validation_predictions.jsonl")
    p.add_argument("--k2-lora",default="cache/results/k2_lora_validation_predictions.jsonl")
    p.add_argument("--num-bootstrap",type=int,default=10_000);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--json-output",default="cache/results/k2_lora_bootstrap.json");p.add_argument("--csv-output",default="cache/results/k2_lora_bootstrap.csv");p.add_argument("--markdown-output",default="cache/results/k2_lora_bootstrap.md");return p.parse_args()


def main():
    a=parse_args();data={"K1_BASE":load(a.k1_base,variant="V1_TITLE_SENTENCE"),"K1_LORA":load(a.k1_lora,"K1_LORA"),"K2_BASE":load(a.k2_base),"K2_LORA":load(a.k2_lora,"K2_LORA")}
    result=analyze(data,a.num_bootstrap,a.seed);Path(a.json_output).parent.mkdir(parents=True,exist_ok=True);Path(a.json_output).write_text(json.dumps(result,indent=2)+"\n")
    fields=["Comparison","Delta","CI Low","CI High","P(delta>0)","P(delta>=1)","P(delta>=3)"]
    with Path(a.csv_output).open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for name,row in result["comparisons"].items():writer.writerow(dict(zip(fields,[name,row["delta"],*row["delta_ci95"],row["probability_positive"],row["probability_at_least_1"],row["probability_at_least_3"]])))
    lines=["# K2 + LoRA Factorial Paired Bootstrap","",f"- Samples: 500; resamples: {a.num_bootstrap}; seed: {a.seed}.",""]
    for name,row in result["comparisons"].items():lines.append(f"- {name}: {row['delta']:.4f} (95% CI {row['delta_ci95'][0]:.4f} to {row['delta_ci95'][1]:.4f}); P(delta>0)={row['probability_positive']:.4f}.")
    interaction=result["interaction"];lines+= ["",f"- Interaction: {interaction['point_estimate']:.4f} (95% CI {interaction['ci95'][0]:.4f} to {interaction['ci95'][1]:.4f}); P(>0)={interaction['probability_positive']:.4f}; P(<0)={interaction['probability_negative']:.4f}."]
    Path(a.markdown_output).write_text("\n".join(lines)+"\n");print(json.dumps(result,indent=2))


if __name__=="__main__":main()
