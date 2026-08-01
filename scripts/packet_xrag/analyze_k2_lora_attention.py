#!/usr/bin/env python
"""Diagnose K2 token geometry and early-layer attention for Gate C/D."""

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F
from transformers import AutoConfig,AutoTokenizer

REPO_ROOT=Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:sys.path.insert(0,str(REPO_ROOT))

from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag import train_residual_packet_projector as locked
from scripts.packet_xrag.run_k2_lora_combination import build_model,parse_args as combination_args
from src.language_modeling.utils import XRAG_TOKEN
from src.model import SFR,XMistralForCausalLM
from src.packet_xrag.encoding.multi_token_projector import MultiTokenPacketProjector
from src.packet_xrag.modeling.multi_token_xrag import install_multi_token_injection


def load_base(args,device,xrag_id):
    config=AutoConfig.from_pretrained(v1.XRAG_MODEL_NAME);config._attn_implementation="eager"
    model=XMistralForCausalLM.from_pretrained(v1.XRAG_MODEL_NAME,config=config,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to(device);model.set_xrag_token_id(xrag_id)
    model.projector.load_state_dict(torch.load(args.base_projector_checkpoint,map_location="cpu",weights_only=True));model.projector=MultiTokenPacketProjector(model.projector,4096,4096,2,1024).to(device=device,dtype=torch.bfloat16)
    model.projector.load_state_dict(torch.load(args.k2_projector_checkpoint,map_location="cpu",weights_only=True));install_multi_token_injection(model);model.eval()
    for p in model.parameters():p.requires_grad=False
    return model


@torch.inference_mode()
def collect(model,configuration,records,tokenizer,sfr_tokenizer,sfr,device,xrag_id):
    rows=[]
    for sample,gold,_ in records:
        prompt=v1.build_prompt(sample["question"],len(gold)*2);tokens=tokenizer(prompt,return_tensors="pt",add_special_tokens=False).to(device)
        retrieval=v1.encode_packets(sfr_tokenizer,sfr,[p["encoder_text"] for p in gold],device);projected=model.projector(retrieval).float()
        token1,token2=projected[:,0],projected[:,1];positions=(tokens.input_ids[0]==xrag_id).nonzero().flatten();assert len(positions)==len(gold)*2
        output=model(input_ids=tokens.input_ids,attention_mask=tokens.attention_mask,retrieval_embeds=retrieval,output_attentions=True,use_cache=False)
        row={"sample_id":locked.sample_id(sample),"configuration":configuration,"num_packets":len(gold),
            "first_soft_token_norm":float(token1.norm(dim=-1).mean()),"second_soft_token_norm":float(token2.norm(dim=-1).mean()),
            "cosine_token1_token2":float(F.cosine_similarity(token1,token2,dim=-1).mean())}
        masses1=[];masses2=[]
        for layer in range(4):
            selected=output.attentions[layer][0,:,-1,positions].float().mean(dim=0).reshape(len(gold),2)
            first=float(selected[:,0].mean());second=float(selected[:,1].mean());row[f"layer_{layer}_token1_attention"]=first;row[f"layer_{layer}_token2_attention"]=second
            row[f"layer_{layer}_token2_token1_ratio"]=second/max(first,1e-12);masses1.append(first);masses2.append(second)
        row["attention_mass_token1"]=mean(masses1);row["attention_mass_token2"]=mean(masses2);row["attention_mass_ratio_token2_token1"]=row["attention_mass_token2"]/max(row["attention_mass_token1"],1e-12);rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--max-samples",type=int,default=100);p.add_argument("--output",default="cache/results/k2_lora_attention_diagnostic.csv");p.add_argument("--markdown-output",default="cache/results/k2_lora_attention_diagnostic.md");p.add_argument("--device",default="cuda:0");args=p.parse_args();assert args.max_samples==100
    defaults=combination_args([]);defaults.configuration="K2_LORA";defaults.device=args.device
    device=torch.device(args.device);torch.cuda.set_device(device);tokenizer=AutoTokenizer.from_pretrained(v1.XRAG_MODEL_NAME,padding_side="left",add_eos_token=False,use_fast=False)
    if tokenizer.pad_token_id is None:tokenizer.pad_token_id=tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id
    xrag_id=tokenizer.convert_tokens_to_ids(XRAG_TOKEN);sfr_tokenizer=AutoTokenizer.from_pretrained(v1.SFR_MODEL_NAME);sfr=SFR.from_pretrained(v1.SFR_MODEL_NAME,torch_dtype=torch.bfloat16).eval().to(device)
    defaults.data_split=defaults.split_file;defaults.train_samples=5000;defaults.validation_samples=500;defaults.seed=42;defaults.max_packets=4;records=locked.load_locked_split(defaults)[1][:100]
    base=load_base(defaults,device,xrag_id);rows=collect(base,"K2_BASE",records,tokenizer,sfr_tokenizer,sfr,device,xrag_id);del base;torch.cuda.empty_cache()
    combo,_,_=build_model(defaults,device,tokenizer,xrag_id);rows+=collect(combo,"K2_LORA",records,tokenizer,sfr_tokenizer,sfr,device,xrag_id)
    fields=list(rows[0]);Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    with Path(args.output).open("w",newline="") as stream:writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    by={name:[row for row in rows if row["configuration"]==name] for name in ("K2_BASE","K2_LORA")};summary={name:{key:mean(row[key] for row in items) for key in fields if key not in {"sample_id","configuration"}} for name,items in by.items()}
    lines=["# K2 + LoRA Attention Diagnostic","","Fixed first 100 examples of the locked projector-validation split; this diagnostic does not alter Gate C.",""]
    for name in ("K2_BASE","K2_LORA"):
        s=summary[name];lines += [f"## {name}","",f"- Token 1 norm: {s['first_soft_token_norm']:.6f}",f"- Token 2 norm: {s['second_soft_token_norm']:.6f}",f"- Cosine(token1, token2): {s['cosine_token1_token2']:.6f}",f"- Attention token1: {s['attention_mass_token1']:.8f}",f"- Attention token2: {s['attention_mass_token2']:.8f}",f"- Token2/token1 ratio: {s['attention_mass_ratio_token2_token1']:.6f}",""]
    Path(args.markdown_output).write_text("\n".join(lines));print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
