#!/usr/bin/env python
"""Frozen full-validation composition of the best K=2 projector and existing LoRA."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import run_selector_calibration as selector
from scripts.packet_xrag import train_multi_token_projector as multi_train
from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag import train_residual_packet_projector as locked
from src.language_modeling.utils import XRAG_TOKEN
from src.model import SFR, XMistralForCausalLM
from src.packet_xrag.encoding.multi_token_projector import MultiTokenPacketProjector
from src.packet_xrag.encoding.residual_projector import ResidualPacketProjector
from src.packet_xrag.modeling.multi_token_xrag import install_multi_token_injection

EXPECTED_SPLIT_HASH="8f925ff8ababf1efc6bb8a913e6d5431437610b0bb30fa8357a57dfbb5f24052"


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__(); self.base=base; self.scaling=alpha/rank; self.dropout=nn.Dropout(dropout)
        self.lora_a=nn.Linear(base.in_features,rank,bias=False,device=base.weight.device,dtype=base.weight.dtype)
        self.lora_b=nn.Linear(rank,base.out_features,bias=False,device=base.weight.device,dtype=base.weight.dtype)
        for parameter in self.base.parameters(): parameter.requires_grad=False
    def forward(self,inputs): return self.base(inputs)+self.lora_b(self.lora_a(self.dropout(inputs)))*self.scaling


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def install_lora(model, config):
    installed=[]
    for layer_index in config["selected_layers"]:
        attention=model.model.layers[layer_index].self_attn
        for target in config["target_modules"]:
            base=getattr(attention,target)
            setattr(attention,target,LoRALinear(base,config["rank"],config["lora_alpha"],config["lora_dropout"]))
            installed.append(f"model.layers.{layer_index}.self_attn.{target}")
    assert installed==config["replaced_modules"]
    return installed


def load_lora_state(model,path):
    state=torch.load(path,map_location="cpu",weights_only=True)
    missing,unexpected=model.load_state_dict(state,strict=False)
    assert not unexpected
    loaded={name for name in model.state_dict() if ".lora_a." in name or ".lora_b." in name}
    assert loaded==set(state)
    return state,missing


def validate_lora_targets(model, expected):
    actual=[]
    for name,module in model.named_modules():
        if isinstance(module,LoRALinear): actual.append(name)
    assert actual==expected
    assert all(".self_attn." in name and (name.endswith("q_proj") or name.endswith("v_proj")) for name in actual)
    return actual


def build_model(args,device,tokenizer,xrag_id):
    config=AutoConfig.from_pretrained(v1.XRAG_MODEL_NAME)
    model=XMistralForCausalLM.from_pretrained(v1.XRAG_MODEL_NAME,config=config,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to(device)
    model.set_xrag_token_id(xrag_id)
    model.projector.load_state_dict(torch.load(args.base_projector_checkpoint,map_location="cpu",weights_only=True),strict=True)
    if args.configuration=="K2_LORA":
        model.projector=MultiTokenPacketProjector(model.projector,config.retriever_hidden_size,config.hidden_size,2,1024).to(device=device,dtype=torch.bfloat16)
        model.projector.load_state_dict(torch.load(args.k2_projector_checkpoint,map_location="cpu",weights_only=True),strict=True)
        install_multi_token_injection(model); k=2
    else:
        residual=ResidualPacketProjector(model.projector,hidden_size=config.hidden_size,bottleneck_size=1024).to(device=device,dtype=torch.bfloat16)
        residual.load_state_dict(torch.load(args.lora_projector_checkpoint,map_location="cpu",weights_only=True),strict=True)
        model.projector=residual; k=1
    lora_config=json.loads(Path(args.lora_training_config).read_text())
    expected=install_lora(model,lora_config); load_lora_state(model,args.lora_checkpoint); validate_lora_targets(model,expected)
    model.eval()
    for parameter in model.parameters():parameter.requires_grad=False
    assert not any(parameter.requires_grad for parameter in model.parameters())
    if k==2: assert isinstance(model.projector,MultiTokenPacketProjector) and model.projector.tokens_per_packet==2
    return model,k,lora_config


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration",choices=["K2_LORA","K1_LORA"],default="K2_LORA")
    parser.add_argument("--max-samples",type=int,default=500)
    parser.add_argument("--validation-ids-file",default="cache/results/packet_representation_ablation_validation_ids.json")
    parser.add_argument("--split-file",default="cache/projector/packet_projector_calibration/data_split.json")
    parser.add_argument("--v1-training-config",default="cache/projector/packet_projector_calibration/last/training_config.json")
    parser.add_argument("--base-projector-checkpoint",default="cache/projector/packet_projector_calibration/last/projector.pt")
    parser.add_argument("--k2-projector-checkpoint",default="cache/projector/multi_token_k2/best_short_f1/multi_token_projector.pt")
    parser.add_argument("--lora-checkpoint",default="cache/lora/early_layer_lora/best_short_f1/adapter.pt")
    parser.add_argument("--lora-training-config",default="cache/lora/early_layer_lora/best_short_f1/training_config.json")
    parser.add_argument("--lora-projector-checkpoint",default="cache/projector/residual_packet_projector/best_short_f1/projector.pt")
    parser.add_argument("--device",default="cuda:0");parser.add_argument("--max-new-tokens",type=int,default=32)
    parser.add_argument("--output",default="cache/results/k2_lora_validation_predictions.jsonl")
    parser.add_argument("--summary-output",default="cache/results/k2_lora_validation_summary.json")
    return parser.parse_args(argv)


def substring_score(prediction,gold):
    pred,target=selector.normalize_answer(prediction),selector.normalize_answer(gold)
    return float(bool(pred) and (pred in target or target in pred))


@torch.inference_mode()
def main():
    args=parse_args();assert args.max_samples==500 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ids_record=json.loads(Path(args.validation_ids_file).read_text());ids=ids_record["sample_ids"]
    assert len(ids)==len(set(ids))==500 and ids_record["split_hash"]==EXPECTED_SPLIT_HASH
    split=json.loads(Path(args.split_file).read_text());assert split["validation_sample_ids"]==ids
    device=torch.device(args.device);torch.cuda.set_device(device)
    tokenizer=AutoTokenizer.from_pretrained(v1.XRAG_MODEL_NAME,padding_side="left",add_eos_token=False,use_fast=False)
    if tokenizer.pad_token_id is None:tokenizer.pad_token_id=tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id
    xrag_id=tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
    sfr_tokenizer=AutoTokenizer.from_pretrained(v1.SFR_MODEL_NAME)
    sfr=SFR.from_pretrained(v1.SFR_MODEL_NAME,torch_dtype=torch.bfloat16).eval().to(device)
    for parameter in sfr.parameters():parameter.requires_grad=False
    model,k,lora_config=build_model(args,device,tokenizer,xrag_id)
    args.data_split=args.split_file;args.train_samples=5000;args.validation_samples=500;args.seed=42;args.max_packets=4
    records=locked.load_locked_split(args)[1]
    assert [locked.sample_id(sample) for sample,_,_ in records]==ids
    dataset=(multi_train.MultiTokenDataset(records,tokenizer,xrag_id,42,training=False,tokens_per_packet=2)
             if k==2 else v1.ProjectorDataset(records,tokenizer,xrag_id,42,training=False))
    loader=DataLoader(dataset,batch_size=8,shuffle=False,collate_fn=v1.make_collator(tokenizer))
    nll=v1.validate_loss(model,sfr_tokenizer,sfr,loader,device)
    predictions=[]
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w") as stream:
        for sample,gold,_ in records:
            prompt=v1.build_prompt(sample["question"],len(gold)*k)
            tokenized=tokenizer(prompt,return_tensors="pt",add_special_tokens=False).to(device)
            retrieval=v1.encode_packets(sfr_tokenizer,sfr,[packet["encoder_text"] for packet in gold],device)
            assert int((tokenized.input_ids==xrag_id).sum())==len(gold)*k
            generated=model.generate(input_ids=tokenized.input_ids,attention_mask=tokenized.attention_mask,retrieval_embeds=retrieval,
                do_sample=False,max_new_tokens=args.max_new_tokens,use_cache=True,pad_token_id=tokenizer.pad_token_id)
            new=generated[:,tokenized.input_ids.shape[1]:] if generated.shape[1]>tokenized.input_ids.shape[1] else generated
            raw=tokenizer.batch_decode(new,skip_special_tokens=False)[0];clean=selector.clean_prediction(raw);short=selector.extract_short_answer(raw) or "[EMPTY]"
            em,f1=selector.score_prediction(short,sample["answer"]);_,clean_f1=selector.score_prediction(clean,sample["answer"])
            row={"sample_id":locked.sample_id(sample),"configuration":args.configuration,"question":sample["question"],"gold_answer":sample["answer"],
                "num_packets":len(gold),"tokens_per_packet":k,"total_soft_tokens":len(gold)*k,"raw_prediction":raw,"clean_prediction":clean,
                "short_prediction":short,"short_em":em,"short_f1":f1,"clean_f1":clean_f1,"substring_match":substring_score(short,sample["answer"]),
                "generated_tokens":new.shape[1],"is_empty":short=="[EMPTY]"}
            assert 0<=f1<=1 and em in (0,1) and row["total_soft_tokens"]==k*row["num_packets"]
            predictions.append(row);stream.write(json.dumps(row,ensure_ascii=False)+"\n");stream.flush()
    assert len(predictions)==500 and [row["sample_id"] for row in predictions]==ids
    summary={"configuration":args.configuration,"num_samples":500,"validation_split_hash":EXPECTED_SPLIT_HASH,"tokens_per_packet":k,
        "short_em":100*mean(row["short_em"] for row in predictions),"short_f1":100*mean(row["short_f1"] for row in predictions),
        "clean_f1":100*mean(row["clean_f1"] for row in predictions),"substring":100*mean(row["substring_match"] for row in predictions),
        "validation_nll":nll,"empty_count":sum(row["is_empty"] for row in predictions),"avg_packets":mean(row["num_packets"] for row in predictions),
        "avg_soft_tokens":mean(row["total_soft_tokens"] for row in predictions),"peak_vram_gb":torch.cuda.max_memory_allocated()/1024**3,
        "checkpoints":{"v1":{"path":str(Path(args.base_projector_checkpoint).resolve()),"sha256":sha256(args.base_projector_checkpoint)},
            "k2":{"path":str(Path(args.k2_projector_checkpoint).resolve()),"sha256":sha256(args.k2_projector_checkpoint)},
            "lora":{"path":str(Path(args.lora_checkpoint).resolve()),"sha256":sha256(args.lora_checkpoint)}},
        "lora_actual_config":{key:lora_config[key] for key in ["selected_layers","target_modules","rank","lora_alpha","lora_dropout","projector_checkpoint"]},
        "trainable_parameters":[]}
    Path(args.summary_output).write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
