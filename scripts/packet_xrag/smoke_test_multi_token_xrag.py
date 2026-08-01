#!/usr/bin/env python
"""GPU smoke test for K=1/2/4 packet-major soft-token injection."""

import copy
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import run_selector_calibration as selector
from scripts.packet_xrag import train_packet_projector as v1
from src.language_modeling.utils import XRAG_TOKEN
from src.model import SFR, XMistralForCausalLM
from src.packet_xrag.encoding.multi_token_projector import MultiTokenPacketProjector
from src.packet_xrag.modeling.multi_token_xrag import install_multi_token_injection


@torch.inference_mode()
def main():
    device=torch.device("cuda:0"); torch.cuda.set_device(device); dtype=torch.bfloat16
    sfr_tokenizer=AutoTokenizer.from_pretrained(v1.SFR_MODEL_NAME)
    sfr=SFR.from_pretrained(v1.SFR_MODEL_NAME,torch_dtype=dtype).eval().to(device)
    tokenizer=AutoTokenizer.from_pretrained(v1.XRAG_MODEL_NAME,padding_side="left",add_eos_token=False,use_fast=False)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id=tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id
    config=AutoConfig.from_pretrained(v1.XRAG_MODEL_NAME)
    model=XMistralForCausalLM.from_pretrained(v1.XRAG_MODEL_NAME,config=config,torch_dtype=dtype,low_cpu_mem_usage=True).eval().to(device)
    xrag_id=tokenizer.convert_tokens_to_ids(XRAG_TOKEN); model.set_xrag_token_id(xrag_id)
    model.projector.load_state_dict(torch.load("cache/projector/packet_projector_calibration/last/projector.pt",map_location="cpu",weights_only=True))
    base=copy.deepcopy(model.projector)
    texts=["[France] Paris is the capital of France.","[Germany] Berlin is the capital of Germany."]
    retrieval=v1.encode_packets(sfr_tokenizer,sfr,texts,device)
    base_output=base(retrieval)
    results=[]
    for k in (1,2,4):
        model.projector=MultiTokenPacketProjector(base,config.retriever_hidden_size,config.hidden_size,k,1024).to(device=device,dtype=dtype)
        install_multi_token_injection(model); model.eval()
        projected=model.projector(retrieval)
        assert projected.shape==(2,k,config.hidden_size) and torch.isfinite(projected).all()
        assert torch.allclose(projected[:,0],base_output,atol=1e-5,rtol=1e-5)
        prompt=v1.build_prompt("What is the capital of Germany?",2*k)
        tokenized=tokenizer(prompt,return_tensors="pt",add_special_tokens=False).to(device)
        assert int((tokenized.input_ids==xrag_id).sum())==2*k
        generated=model.generate(input_ids=tokenized.input_ids,attention_mask=tokenized.attention_mask,retrieval_embeds=retrieval,
            do_sample=False,max_new_tokens=32,use_cache=True,pad_token_id=tokenizer.pad_token_id)
        new=generated[:,tokenized.input_ids.shape[1]:] if generated.shape[1]>tokenized.input_ids.shape[1] else generated
        answer=selector.extract_short_answer(tokenizer.batch_decode(new,skip_special_tokens=False)[0]) or "[EMPTY]"
        results.append({"num_packets":2,"K":k,"total_xrag_tokens":2*k,"projected_shape":list(projected.shape),"generated_answer":answer})
    print(json.dumps({"results":results,"peak_vram_gb":torch.cuda.max_memory_allocated()/1024**3},indent=2))


if __name__=="__main__": main()
