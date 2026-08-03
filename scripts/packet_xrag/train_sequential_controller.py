#!/usr/bin/env python
"""Train Stage-2 support-imitation set-conditioned fixed-k controller."""

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

import torch
from transformers import get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.run_sequential_fixed_k_benchmark import rollout_cache
from scripts.packet_xrag.run_static_scorer_benchmark import (
    evaluate_generator, initialize_generator, load_static_scorer, rank_cache, summarize_rows,
)
from scripts.packet_xrag.train_static_scorer import choose_budget, choose_checkpoint, preload_embeddings
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache, sha256_file
from src.packet_xrag.controller.sequential_controller import (
    SequentialPacketController, multi_positive_action_loss,
)

SEED = 20260803


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cache",default="cache/controller/features/train_features")
    p.add_argument("--dev-cache",default="cache/controller/features/internal_dev_features")
    p.add_argument("--static-checkpoint",default="cache/controller/static/best_short_f1/scorer.pt")
    p.add_argument("--k2-training-config",default="cache/projector/multi_token_k2/best_short_f1/training_config.json")
    p.add_argument("--output-dir",default="cache/controller/sequential_fixed_k")
    p.add_argument("--device",default="cuda:1");p.add_argument("--epochs",type=int,default=8)
    p.add_argument("--effective-batch-size",type=int,default=32)
    p.add_argument("--learning-rate",type=float,default=1e-4);p.add_argument("--weight-decay",type=float,default=.01)
    p.add_argument("--warmup-ratio",type=float,default=.05);p.add_argument("--gradient-clipping",type=float,default=1.)
    p.add_argument("--seed",type=int,default=SEED);p.add_argument("--log-every",type=int,default=100)
    return p.parse_args(argv)


def safe_prefix(order, gold):
    selected=[]
    for item in order:
        if set(selected+[item]) >= set(gold): break
        selected.append(item)
    return selected


def trajectory_states(record, static_ranking, epoch, seed=SEED):
    gold=list(record["gold_packet_ids"]); topk=record["topk_ranking"]
    states=[]
    for trajectory in range(4):
        order=gold.copy();random.Random(f"{seed}:{epoch}:{record['sample_id']}:{trajectory}").shuffle(order)
        states.append(("gold", order[:trajectory % len(order)]))
    for prefix in (1,2): states.append(("topk",safe_prefix(topk[:prefix],gold)))
    for prefix in (1,2): states.append(("static",safe_prefix(static_ranking[:prefix],gold)))
    support_docs={record["packets"][i]["doc_id"] for i in gold}
    same=next((i for i,p in enumerate(record["packets"]) if i not in gold and p["doc_id"] in support_docs),None)
    distractor=next((i for i,p in enumerate(record["packets"]) if i not in gold and p["doc_id"] not in support_docs),None)
    states.append(("same_document",[] if same is None else [same]))
    states.append(("random_error",[] if distractor is None else [distractor]))
    assert [x[0] for x in states].count("gold")==4 and len(states)==10
    assert all(not set(selected)>=set(gold) for _,selected in states)
    return states


def state_loss(model, record, selected, device):
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
        remaining,scores=model.score_record(record,selected,device)
    positives=torch.tensor([i in set(record["gold_packet_ids"]) for i in remaining],device=device)
    return multi_positive_action_loss(scores.float(),positives)


@torch.inference_mode()
def dev_action_loss(model,cache,static_rankings,device):
    losses=[]
    for i in range(len(cache)):
        record=cache[i]
        for _,selected in trajectory_states(
                record, static_rankings[record["sample_id"]], 0):
            losses.append(float(state_loss(model,record,selected,device)))
    return sum(losses)/len(losses)


def save(model,directory,metadata):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},directory/"controller.pt")
    (directory/"checkpoint_metadata.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n")


def main(argv=None):
    a=parse_args(argv)
    locked=(a.epochs,a.effective_batch_size,a.learning_rate,a.weight_decay,a.warmup_ratio,a.gradient_clipping,a.seed)
    if locked!=(8,32,1e-4,.01,.05,1.,SEED):raise RuntimeError("sequential protocol mismatch")
    random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    device=torch.device(a.device);torch.cuda.set_device(device)
    train=ControllerFeatureCache(a.train_cache);dev=ControllerFeatureCache(a.dev_cache)
    preload_embeddings(train,device);preload_embeddings(dev,device)
    static=load_static_scorer(a.static_checkpoint,device)
    static_rank=rank_cache(train,static,device);static_dev_rank=rank_cache(dev,static,device)
    del static
    model=SequentialPacketController().to(device)
    static_state=torch.load(a.static_checkpoint,map_location="cpu",weights_only=True);model.initialize_from_static(static_state)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay,fused=True)
    states_per_epoch=len(train)*10;updates=math.ceil(states_per_epoch/32);total=updates*8
    scheduler=get_linear_schedule_with_warmup(optimizer,int(total*.05),total)
    tokenizer,generator,xrag_id,hashes=initialize_generator(a.k2_training_config,device)
    history=[];out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,9):
        states=[]
        for i,record in enumerate(train.records):
            states.extend((i,category,selected) for category,selected in trajectory_states(record,static_rank[record["sample_id"]],epoch))
        random.Random(a.seed+epoch).shuffle(states);model.train();losses=[]
        for batch,start in enumerate(range(0,len(states),32),1):
            optimizer.zero_grad(set_to_none=True);items=states[start:start+32]
            batch_losses=[state_loss(model,train[i],selected,device) for i,_,selected in items]
            loss=torch.stack(batch_losses).mean();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();scheduler.step();losses.append(float(loss.detach()))
            if batch%a.log_every==0 or batch==updates:print(json.dumps({"epoch":epoch,"batch":batch,"batches":updates,"loss":losses[-1],"gradient_norm":float(norm)}),flush=True)
        action_loss=dev_action_loss(model,dev,static_dev_rank,device)
        rankings=rollout_cache(dev,model,device)
        epoch_dir=out/f"epoch_{epoch}";pred=epoch_dir/"validation_predictions.jsonl"
        rows=evaluate_generator(dev,rankings,tokenizer,generator,xrag_id,device,pred)
        for row in rows: row["configuration"]=row["configuration"].replace("STATIC","SEQ")
        with pred.open("w") as stream:
            for row in rows: stream.write(json.dumps(row,ensure_ascii=False)+"\n")
        metrics={n.replace("STATIC","SEQ"):v for n,v in summarize_rows(rows).items()}
        budget=choose_budget({n.replace("SEQ","STATIC"):v for n,v in metrics.items()})
        rec={"epoch":epoch,"training_action_loss":sum(losses)/len(losses),"validation_action_loss":action_loss,
             "selected_budget":budget,"selected_short_f1":metrics[f"SEQ_{budget}"]["short_f1"],"metrics":metrics}
        history.append(rec);save(model,epoch_dir,rec)
        (epoch_dir/"validation_metrics.json").write_text(json.dumps(rec,indent=2,sort_keys=True)+"\n");print(json.dumps(rec,indent=2),flush=True)
    # Same global selection policy with action loss as second tie.
    maximum=max(x["selected_short_f1"] for x in history);close=[x for x in history if maximum-x["selected_short_f1"]<.25]
    selected=min(close,key=lambda x:(x["selected_budget"],x["validation_action_loss"],x["epoch"]))
    for destination,source in ((out/"best_short_f1",out/f"epoch_{selected['epoch']}"),(out/"last",out/"epoch_8")):
        destination.mkdir(parents=True,exist_ok=True)
        for f in ("controller.pt","checkpoint_metadata.json","validation_metrics.json","validation_predictions.jsonl"):shutil.copyfile(source/f,destination/f)
    config={"stage":"sequential_fixed_k","epochs":8,"effective_batch_size":32,"learning_rate":1e-4,"weight_decay":.01,
            "warmup_ratio":.05,"gradient_clipping":1.,"dtype":"BF16 autocast","seed":SEED,
            "trajectory_ratio":{"gold":.4,"topk":.2,"static":.2,"same_document":.1,"random_error":.1},
            "initialization":"static Gate A query/packet projections; candidate MLP newly initialized",
            "static_checkpoint":str(Path(a.static_checkpoint).resolve()),"static_sha256":sha256_file(a.static_checkpoint),
            "selected_epoch":selected["epoch"],"selected_budget":selected["selected_budget"],"history":history,
            "generator_checkpoint_hashes":hashes,"generator_trainable_parameters":0,"benchmark_accessed_during_training":False,
            "final_100_accessed":False,"final_100_runs":0}
    for d in (out/"best_short_f1",out/"last"):(d/"training_config.json").write_text(json.dumps(config,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"selected":selected},indent=2),flush=True)


if __name__=="__main__":main()
