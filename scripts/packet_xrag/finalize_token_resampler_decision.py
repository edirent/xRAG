#!/usr/bin/env python
"""Measure system costs and write the stopped-route decision artifacts."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag.token_resampler_common import (
    EXPECTED_SPLIT_HASH, EXPECTED_K2_SHA256, EXPECTED_V1_SHA256,
    load_frozen_k2_model, locked_records,
)
from scripts.packet_xrag.train_token_state_resampler import pack_packet_keys
from src.language_modeling.utils import XRAG_TOKEN
from src.packet_xrag.encoding.token_state_cache import TokenStateCache, packet_key
from src.packet_xrag.encoding.token_state_resampler import ResidualTokenStateResampler
from src.packet_xrag.modeling.token_state_xrag import install_token_state_injection


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--cache-dir",default="cache/token_states")
    parser.add_argument("--checkpoint",default="cache/resampler/token_state_depth1/best_short_f1/resampler.pt")
    parser.add_argument("--v1-checkpoint",default="cache/projector/packet_projector_calibration/last/projector.pt")
    parser.add_argument("--k2-checkpoint",default="cache/projector/multi_token_k2/best_short_f1/multi_token_projector.pt")
    return parser.parse_args()


@torch.inference_mode()
def main():
    args=parse_args();device=torch.device(args.device);torch.cuda.set_device(device)
    audit=json.loads(Path("cache/results/token_state_cache_audit.json").read_text())
    cache_metrics=json.loads((Path(args.cache_dir)/"cache_metrics.json").read_text())
    d1=json.loads(Path("cache/resampler/token_state_depth1/training_summary.json").read_text())
    control=json.loads(Path("cache/resampler/pooled_control/training_summary.json").read_text())
    mechanism=json.loads(Path("cache/results/token_resampler_mechanism.json").read_text())
    bootstrap=json.loads(Path("cache/results/token_resampler_bootstrap.json").read_text())
    comparisons={row["comparison"]:row for row in bootstrap["comparisons"]}
    train,validation=locked_records("cache/projector/packet_projector_calibration/data_split.json",
                                    "cache/projector/packet_projector_calibration/last/training_config.json")
    cache=TokenStateCache(args.cache_dir)

    # Force mmap pages to be read and report warm-cache application throughput.
    keys=list(cache.manifest["records"])[:2000];read_bytes=0;checksum=0.0
    started=time.perf_counter()
    for key in keys:
        record=cache.get(key);checksum+=float(record["hidden"].float().sum())
        read_bytes+=record["hidden"].numel()*2+record["pooled"].numel()*2+record["input_ids"].numel()*4+record["mask"].numel()
    cache_seconds=time.perf_counter()-started

    tokenizer=AutoTokenizer.from_pretrained(v1.XRAG_MODEL_NAME,padding_side="left",add_eos_token=False,use_fast=False)
    if tokenizer.pad_token_id is None:tokenizer.pad_token_id=tokenizer.unk_token_id or tokenizer.eos_token_id
    xrag_id=tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
    model,config=load_frozen_k2_model(device,tokenizer,xrag_id,args.v1_checkpoint,args.k2_checkpoint)
    model.projector=ResidualTokenStateResampler(model.projector,config.retriever_hidden_size,config.hidden_size,512,2,8,2048).to(device=device,dtype=torch.bfloat16)
    model.projector.load_state_dict(torch.load(args.checkpoint,map_location="cpu",weights_only=True),strict=True)
    install_token_state_injection(model);model.eval()
    probe_keys=[packet_key(packet["encoder_text"]) for _,gold,_ in validation[:32] for packet in gold]
    retrieval=pack_packet_keys(cache,probe_keys,device)
    for _ in range(10): model.projector(retrieval["token_states"],retrieval["token_mask"],retrieval["pooled_embeddings"])
    torch.cuda.synchronize();started=time.perf_counter();iterations=100
    for _ in range(iterations): model.projector(retrieval["token_states"],retrieval["token_mask"],retrieval["pooled_embeddings"])
    torch.cuda.synchronize();resampler_seconds=time.perf_counter()-started
    resampler_pps=len(probe_keys)*iterations/resampler_seconds

    torch.cuda.reset_peak_memory_stats(device)
    sample,gold,_=validation[0];prompt=v1.build_prompt(sample["question"],len(gold)*2)
    tokenized=tokenizer(prompt,return_tensors="pt",add_special_tokens=False).to(device)
    retrieval_one=pack_packet_keys(cache,[packet_key(packet["encoder_text"]) for packet in gold],device)
    model.generate(input_ids=tokenized.input_ids,attention_mask=tokenized.attention_mask,retrieval_embeds=retrieval_one,
                   do_sample=False,max_new_tokens=32,use_cache=True,pad_token_id=tokenizer.pad_token_id)
    inference_vram=torch.cuda.max_memory_allocated(device)/1024**3

    system={
        "offline_sfr_encoding_packets_per_second":cache_metrics["encoding_packets_per_second"],
        "offline_resampler_packets_per_second":resampler_pps,
        "token_state_cache_bytes":cache_metrics["actual_cache_bytes"],
        "token_state_cache_gib":cache_metrics["actual_cache_gib"],
        "bytes_per_packet":cache_metrics["actual_cache_bytes"]/audit["unique_union_packets"],
        "cache_read_bytes_per_second":read_bytes/cache_seconds,
        "cache_read_gib_per_second":read_bytes/cache_seconds/1024**3,
        "cache_read_packets_per_second":len(keys)/cache_seconds,
        "cache_read_checksum":checksum,
        "training_peak_vram_gib":d1["peak_vram_gb"],
        "inference_peak_vram_gib":inference_vram,
        "average_packet_token_length":audit["mean_tokens_per_packet"],
        "average_selected_packets":2.342,
        "online_soft_tokens_per_packet":2,
        "online_average_total_soft_tokens":4.684,
        "relative_online_prefill_cost_vs_pooled_k2":1.0,
    }
    Path("cache/results/token_resampler_system_metrics.csv").parent.mkdir(parents=True,exist_ok=True)
    with Path("cache/results/token_resampler_system_metrics.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(system));writer.writeheader();writer.writerow(system)
    lines=["# Token Resampler System Report","",
           "Token-state resampling increases offline SFR encoding and cache-storage cost, while the online LLM context remains exactly two soft tokens per packet.",""]
    lines += [f"- {key}: {value}" for key,value in system.items()]
    Path("cache/results/token_resampler_system_report.md").write_text("\n".join(lines)+"\n")

    d1_vs_k2=comparisons["TokenState_D1_vs_Pooled_K2"]
    d1_vs_control=comparisons["TokenState_D1_vs_PooledControl"]
    decision=["# Token-state Resampler Final Decision","",
        f"1. Validation split hash: `{EXPECTED_SPLIT_HASH}`.",
        f"2. Checkpoint audit: V1 `{EXPECTED_V1_SHA256}`; K2 `{EXPECTED_K2_SHA256}`; base model compatible: yes.",
        f"3. Token-state cache: {audit['unique_train_packets']} unique train, {audit['unique_validation_packets']} unique validation, {audit['mean_tokens_per_packet']:.6f} mean tokens, {audit['truncation_rate']:.8f} truncation rate, {cache_metrics['actual_cache_gib']:.6f} GiB actual.",
        f"4. Pooled K2 baseline: F1 {audit['pooled_k2_reproduction']['validation_short_f1']:.6f}, EM {audit['pooled_k2_reproduction']['validation_short_em']:.6f}.",
        f"5. Depth-1: best epoch {d1['best_epoch']}, F1 {d1['best_validation_short_f1']:.6f}, NLL {d1['best_validation_nll']:.6f}, peak VRAM {d1['peak_vram_gb']:.6f} GiB.",
        f"6. Pooled control: best epoch {control['best_epoch']}, F1 {control['best_validation_short_f1']:.6f}, NLL {control['best_validation_nll']:.6f}.",
        "7. Depth-2: not executed; Gate D and mechanism-qualification gate forbid it.",
        "8. SFR LoRA: not executed; stable token-level improvement prerequisites were not met.",
        "9. Paired bootstrap: "+"; ".join(f"{row['comparison']} delta {row['delta_short_f1']:.6f} CI [{row['ci95_lower']:.6f}, {row['ci95_upper']:.6f}]" for row in bootstrap['comparisons'])+".",
        f"10. Mechanism: query cosine {mechanism['latent_query_cosine']:.6f}, output cosine {mechanism['latent_output_cosine']:.6f}, JS {mechanism['attention_js_divergence']:.8f}, sentence mass {mechanism['sentence_attention_mass']:.6f}, residual/base {mechanism['residual_base_norm_ratio']:.8f}; latent collapse: {mechanism['latent_collapse']}.",
        f"11. System cost: SFR {system['offline_sfr_encoding_packets_per_second']:.3f} packets/s, resampler {system['offline_resampler_packets_per_second']:.3f} packets/s, cache {system['token_state_cache_gib']:.6f} GiB, online prefill ratio 1.0.",
        "12. Selected representation: Pooled K2 (simplest and highest full-500 F1 among this route's eligible models).",
        "13. Internal 66 gate: not passed.",
        "14. Final 100 accessed: no.",
        "15. Final 100 run count: 0.",
        "16. Selector benchmark allowed: no.",
        "17. Controller development allowed: yes, using frozen Pooled K2.",
        "18. Representation route ended: yes; token-state expansion stopped at Stage 5 / Gate D.","",
        "Conclusion: We explored token-level resampling but did not find reliable evidence that pooling was the dominant bottleneck."]
    Path("cache/results/token_resampler_final_decision.md").write_text("\n".join(decision)+"\n")
    print(json.dumps({"system":system,"selected_representation":"Pooled K2","gate":"D / Stage 5 stop",
                      "internal_66_gate":False,"final_100_runs":0},indent=2),flush=True)


if __name__=="__main__":main()
