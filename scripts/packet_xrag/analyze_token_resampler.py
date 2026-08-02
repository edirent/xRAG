#!/usr/bin/env python
"""Fixed-first-100 attention and latent specialization diagnostics."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag.token_resampler_common import load_frozen_k2_model, locked_records
from scripts.packet_xrag.train_token_state_resampler import pack_packet_keys
from src.language_modeling.utils import XRAG_TOKEN
from src.packet_xrag.encoding.token_state_cache import TokenStateCache, packet_key
from src.packet_xrag.encoding.token_state_resampler import ResidualTokenStateResampler


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--checkpoint",default="cache/resampler/token_state_depth1/best_short_f1/resampler.pt")
    parser.add_argument("--cache-dir",default="cache/token_states")
    parser.add_argument("--split-file",default="cache/projector/packet_projector_calibration/data_split.json")
    parser.add_argument("--v1-training-config",default="cache/projector/packet_projector_calibration/last/training_config.json")
    parser.add_argument("--v1-checkpoint",default="cache/projector/packet_projector_calibration/last/projector.pt")
    parser.add_argument("--k2-checkpoint",default="cache/projector/multi_token_k2/best_short_f1/multi_token_projector.pt")
    parser.add_argument("--output-json",default="cache/results/token_resampler_mechanism.json")
    parser.add_argument("--output-csv",default="cache/results/token_resampler_mechanism.csv")
    parser.add_argument("--output-md",default="cache/results/token_resampler_mechanism.md")
    return parser.parse_args()


def entropy(probabilities):
    values=probabilities.clamp_min(1e-9)
    return float(-(values*values.log()).sum())


def js_divergence(left,right):
    middle=(left+right)/2
    return float(0.5*((left.clamp_min(1e-9)*(left.clamp_min(1e-9).log()-middle.clamp_min(1e-9).log())).sum()
                      +(right.clamp_min(1e-9)*(right.clamp_min(1e-9).log()-middle.clamp_min(1e-9).log())).sum()))


@torch.inference_mode()
def main():
    args=parse_args();device=torch.device(args.device);torch.cuda.set_device(device)
    cache=TokenStateCache(args.cache_dir)
    records=locked_records(args.split_file,args.v1_training_config)[1][:100]
    tokenizer=AutoTokenizer.from_pretrained(v1.XRAG_MODEL_NAME,padding_side="left",add_eos_token=False,use_fast=False)
    if tokenizer.pad_token_id is None:tokenizer.pad_token_id=tokenizer.unk_token_id or tokenizer.eos_token_id
    xrag_id=tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
    model,config=load_frozen_k2_model(device,tokenizer,xrag_id,args.v1_checkpoint,args.k2_checkpoint)
    k2=model.projector
    model.projector=ResidualTokenStateResampler(k2,config.retriever_hidden_size,config.hidden_size,512,2,8,2048).to(device=device,dtype=torch.bfloat16)
    model.projector.load_state_dict(torch.load(args.checkpoint,map_location="cpu",weights_only=True),strict=True);model.projector.eval()
    rows=[]
    for sample,gold,_ in records:
        keys=[packet_key(packet["encoder_text"]) for packet in gold]
        retrieval=pack_packet_keys(cache,keys,device)
        output,diagnostic=model.projector(**{
            "token_states":retrieval["token_states"],"token_mask":retrieval["token_mask"],
            "pooled_embeddings":retrieval["pooled_embeddings"],"return_diagnostics":True})
        attention=diagnostic["attention"].float().mean(dim=1)
        for packet_index,key in enumerate(keys):
            metadata=cache.get(key)["metadata"];length=metadata["length"]
            weights=attention[packet_index,:,:length]
            weights=weights/weights.sum(dim=-1,keepdim=True)
            title=metadata["title_span"];sentence=metadata["sentence_span"]
            title_mass=float(weights[:,title[0]:title[1]].sum(dim=-1).mean()) if title[1]>title[0] else 0.0
            sentence_mass=float(weights[:,sentence[0]:sentence[1]].sum(dim=-1).mean()) if sentence[1]>sentence[0] else 0.0
            special_mass=float(weights[:,metadata["special_positions"]].sum(dim=-1).mean()) if metadata["special_positions"] else 0.0
            residual=diagnostic["residual"][packet_index].float();base=diagnostic["base"][packet_index].float()
            rows.append({"sample_id":str(sample["id"]),"packet_key":key,
                "latent_output_cosine":float(F.cosine_similarity(output[packet_index,0].float(),output[packet_index,1].float(),dim=0)),
                "residual_cosine":float(F.cosine_similarity(residual[0],residual[1],dim=0)),
                "attention_js_divergence":js_divergence(weights[0],weights[1]),
                "attention_entropy_latent_1":entropy(weights[0]),"attention_entropy_latent_2":entropy(weights[1]),
                "title_attention_mass":title_mass,"sentence_attention_mass":sentence_mass,"special_attention_mass":special_mass,
                "residual_base_norm_ratio":float(residual.norm()/(base.norm()+1e-9))})
    query_cosine=float(F.cosine_similarity(model.projector.latent_queries[0].float(),model.projector.latent_queries[1].float(),dim=0))
    summary={"examples":100,"packets":len(rows),"latent_query_cosine":query_cosine}
    for field in ["latent_output_cosine","residual_cosine","attention_js_divergence","attention_entropy_latent_1","attention_entropy_latent_2","title_attention_mass","sentence_attention_mass","special_attention_mass","residual_base_norm_ratio"]:
        summary[field]=mean(row[field] for row in rows)
    summary["latent_collapse"]=(summary["attention_js_divergence"]<1e-3 or abs(summary["latent_output_cosine"])>0.99)
    Path(args.output_json).write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    with Path(args.output_csv).open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    Path(args.output_md).write_text("# Token Resampler Mechanism Diagnostic\n\n"+"\n".join(f"- {key}: {value}" for key,value in summary.items())+"\n")
    print(json.dumps(summary,indent=2,sort_keys=True),flush=True)


if __name__=="__main__":main()
