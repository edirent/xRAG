import torch
from torch import nn

from scripts.packet_xrag.run_k2_lora_combination import install_lora, validate_lora_targets
from src.packet_xrag.encoding.multi_token_projector import MultiTokenPacketProjector


class Attention(nn.Module):
    def __init__(self):
        super().__init__(); self.q_proj=nn.Linear(4,4,bias=False); self.v_proj=nn.Linear(4,4,bias=False); self.k_proj=nn.Linear(4,4,bias=False); self.o_proj=nn.Linear(4,4,bias=False)


class Toy(nn.Module):
    def __init__(self):
        super().__init__(); self.model=nn.Module(); self.model.layers=nn.ModuleList()
        for _ in range(3):
            layer=nn.Module();layer.self_attn=Attention();layer.mlp=nn.Linear(4,4);self.model.layers.append(layer)
        self.projector=MultiTokenPacketProjector(nn.Linear(3,4),3,4,2,2)


def config():
    modules=[f"model.layers.{i}.self_attn.{target}" for i in (0,1) for target in ("q_proj","v_proj")]
    return {"selected_layers":[0,1],"target_modules":["q_proj","v_proj"],"rank":2,"lora_alpha":4.,"lora_dropout":0.,"replaced_modules":modules}


def test_k2_checkpoint_round_trip_is_exact():
    source=Toy().projector; target=MultiTokenPacketProjector(nn.Linear(3,4),3,4,2,2)
    target.load_state_dict(source.state_dict()); x=torch.randn(2,3)
    assert torch.allclose(source(x),target(x),atol=1e-5,rtol=1e-5)


def test_lora_targets_effect_and_no_accidental_modules():
    torch.manual_seed(1); model=Toy(); x=torch.randn(2,4)
    before=model.model.layers[0].self_attn.q_proj(x)
    expected=install_lora(model,config());validate_lora_targets(model,expected)
    with torch.no_grad():model.model.layers[0].self_attn.q_proj.lora_a.weight.fill_(.1);model.model.layers[0].self_attn.q_proj.lora_b.weight.fill_(.2)
    after=model.model.layers[0].self_attn.q_proj(x)
    assert not torch.allclose(before,after,atol=1e-7,rtol=1e-7)
    assert not hasattr(model.projector,"lora_a") and not hasattr(model.model.layers[0].mlp,"lora_a")


def test_composed_model_can_be_fully_frozen():
    model=Toy();install_lora(model,config())
    for parameter in model.parameters():parameter.requires_grad=False
    assert not any(parameter.requires_grad for parameter in model.parameters())
