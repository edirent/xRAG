#!/usr/bin/env python
"""Smoke test frozen K2 base and K2+LoRA checkpoint composition."""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT=Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:sys.path.insert(0,str(REPO_ROOT))

from scripts.packet_xrag import run_selector_calibration as selector
from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag.run_k2_lora_combination import build_model,sha256
from src.language_modeling.utils import XRAG_TOKEN
from src.model import SFR,XMistralForCausalLM
from src.packet_xrag.encoding.multi_token_projector import MultiTokenPacketProjector
from src.packet_xrag.modeling.multi_token_xrag import install_multi_token_injection


@torch.inference_mode()
def generate(model,tokenizer,xrag_id,retrieval,device):
    prompt=v1.build_prompt("What is the capital of Germany?",4);tokens=tokenizer(prompt,return_tensors="pt",add_special_tokens=False).to(device)
    assert int((tokens.input_ids==xrag_id).sum())==4
    projected=model.projector(retrieval);assert projected.shape==(2,2,4096) and torch.isfinite(projected).all()
    output=model.generate(input_ids=tokens.input_ids,attention_mask=tokens.attention_mask,retrieval_embeds=retrieval,do_sample=False,max_new_tokens=32,use_cache=True,pad_token_id=tokenizer.pad_token_id)
    new=output[:,tokens.input_ids.shape[1]:] if output.shape[1]>tokens.input_ids.shape[1] else output
    return selector.extract_short_answer(tokenizer.batch_decode(new,skip_special_tokens=False)[0]) or "[EMPTY]",projected


@torch.inference_mode()
def main():
    args=argparse.Namespace(configuration="K2_LORA",base_projector_checkpoint="cache/projector/packet_projector_calibration/last/projector.pt",
        k2_projector_checkpoint="cache/projector/multi_token_k2/best_short_f1/multi_token_projector.pt",lora_checkpoint="cache/lora/early_layer_lora/best_short_f1/adapter.pt",
        lora_training_config="cache/lora/early_layer_lora/best_short_f1/training_config.json",lora_projector_checkpoint="cache/projector/residual_packet_projector/best_short_f1/projector.pt")
    device=torch.device("cuda:0");torch.cuda.set_device(device)
    sfr_tokenizer=AutoTokenizer.from_pretrained(v1.SFR_MODEL_NAME);sfr=SFR.from_pretrained(v1.SFR_MODEL_NAME,torch_dtype=torch.bfloat16).eval().to(device)
    tokenizer=AutoTokenizer.from_pretrained(v1.XRAG_MODEL_NAME,padding_side="left",add_eos_token=False,use_fast=False)
    if tokenizer.pad_token_id is None:tokenizer.pad_token_id=tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id
    xrag_id=tokenizer.convert_tokens_to_ids(XRAG_TOKEN);config=AutoConfig.from_pretrained(v1.XRAG_MODEL_NAME)
    retrieval=v1.encode_packets(sfr_tokenizer,sfr,["[France] Paris is the capital of France.","[Germany] Berlin is the capital of Germany."],device)
    base=XMistralForCausalLM.from_pretrained(v1.XRAG_MODEL_NAME,config=config,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to(device);base.set_xrag_token_id(xrag_id)
    base.projector.load_state_dict(torch.load(args.base_projector_checkpoint,map_location="cpu",weights_only=True));base.projector=MultiTokenPacketProjector(base.projector,4096,4096,2,1024).to(device=device,dtype=torch.bfloat16)
    base.projector.load_state_dict(torch.load(args.k2_projector_checkpoint,map_location="cpu",weights_only=True));install_multi_token_injection(base);base.eval()
    base_answer,base_projected=generate(base,tokenizer,xrag_id,retrieval,device);del base;torch.cuda.empty_cache()
    combo,k,lora_config=build_model(args,device,tokenizer,xrag_id);combo_answer,combo_projected=generate(combo,tokenizer,xrag_id,retrieval,device)
    print(json.dumps({"checkpoints":{"v1":[args.base_projector_checkpoint,sha256(args.base_projector_checkpoint)],"k2":[args.k2_projector_checkpoint,sha256(args.k2_projector_checkpoint)],"lora":[args.lora_checkpoint,sha256(args.lora_checkpoint)]},
        "active_adapter":"manual_lora_layers_0_7_q_proj_v_proj","num_packets":2,"tokens_per_packet":k,"total_xrag_tokens":4,"projected_shape":list(combo_projected.shape),
        "K2_BASE_answer":base_answer,"K2_LORA_answer":combo_answer,"projector_unchanged_by_lora":torch.equal(base_projected,combo_projected),
        "finite":bool(torch.isfinite(combo_projected).all()),"peak_vram_gb":torch.cuda.max_memory_allocated()/1024**3,"actual_lora_targets":lora_config["replaced_modules"]},indent=2))


if __name__=="__main__":main()
