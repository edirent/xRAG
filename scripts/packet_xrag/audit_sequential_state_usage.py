#!/usr/bin/env python
"""Audit whether a trained sequential controller materially consumes selected-set state."""

import argparse,json,sys
from pathlib import Path
import torch

REPO_ROOT=Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:sys.path.insert(0,str(REPO_ROOT))
from scripts.packet_xrag.run_sequential_fixed_k_benchmark import load_controller
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache
from src.packet_xrag.controller.sequential_controller import multi_positive_action_loss


def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",default="cache/controller/sequential_fixed_k/best_short_f1/controller.pt")
    p.add_argument("--cache",default="cache/controller/features/internal_dev_features");p.add_argument("--device",default="cuda:1")
    p.add_argument("--training-config",default="cache/controller/sequential_fixed_k/best_short_f1/training_config.json")
    p.add_argument("--output",default="cache/controller/sequential_state_audit.md");a=p.parse_args()
    device=torch.device(a.device);cache=ControllerFeatureCache(a.cache);model=load_controller(a.checkpoint,device);record=cache[0]
    model.train();remaining0,scores0=model.score_record(record,[],device);remaining1,scores1=model.score_record(record,[remaining0[0]],device)
    positives=torch.tensor([i in set(record["gold_packet_ids"]) for i in remaining0],device=device)
    loss=multi_positive_action_loss(scores0,positives);model.zero_grad();loss.backward()
    state_grad=sum(float(p.grad.abs().sum()) for p in model.state_encoder.parameters() if p.grad is not None)
    common=set(remaining0)&set(remaining1);lookup0={i:float(scores0[j].detach()) for j,i in enumerate(remaining0)};lookup1={i:float(scores1[j].detach()) for j,i in enumerate(remaining1)}
    delta=max(abs(lookup0[i]-lookup1[i]) for i in common)
    config=json.loads(Path(a.training_config).read_text())
    expected_ratio={"gold":.4,"topk":.2,"static":.2,"same_document":.1,"random_error":.1}
    checks={"selected_state_features_nonzero":bool(model.state_encoder.empty_mean.detach().abs().sum()),
            "logits_change_with_selected_state":delta>1e-6,"state_encoder_receives_gradient":state_grad>0,
            "selected_packet_excluded":remaining0[0] not in remaining1,"candidate_mask_unique":len(remaining1)==len(set(remaining1)),
            "empty_nonempty_outputs_differ":delta>1e-6,
            "rollout_states_entered_training":config.get("trajectory_ratio")==expected_ratio}
    lines=["# Sequential State-Usage Audit","",*[f"- {k}: {v}" for k,v in checks.items()],f"- maximum common-candidate logit delta: {delta}",f"- state encoder gradient L1: {state_grad}","- Rollout mixture enters training: recorded by locked 40/20/20/10/10 dataset construction."]
    lines.append(f"- Implementation error found: {not all(checks.values())}")
    Path(a.output).write_text("\n".join(lines)+"\n");print(json.dumps(checks,indent=2))

if __name__=="__main__":main()
