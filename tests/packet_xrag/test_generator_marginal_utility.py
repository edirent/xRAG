import re

import pytest
import torch

from src.language_modeling.utils import XRAG_TOKEN
from src.packet_xrag.controller.generator_utility import (
    build_gold_answer_inputs, choose_state_utility_action, delta_utility,
    gold_answer_nll_batch, mean_answer_nll_from_logits, paired_bootstrap,
    static_utility_rollout, utility_cache_key,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self, xrag_id=99):
        self.xrag_id = xrag_id

    def __call__(self, text, add_special_tokens=False):
        pieces = re.split(f"({re.escape(XRAG_TOKEN)})", text)
        ids = []
        for piece in pieces:
            if piece == XRAG_TOKEN:
                ids.append(self.xrag_id)
            else:
                ids.extend(3 + ord(char) % 40 for char in piece)
        return {"input_ids": ids}


class FakeGenerator(torch.nn.Module):
    def __init__(self, vocab_size=128):
        super().__init__()
        self.register_buffer("anchor", torch.zeros(1))
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask, retrieval_embeds=None):
        logits = torch.zeros(*input_ids.shape, self.vocab_size, device=input_ids.device)
        logits.scatter_(2, input_ids.unsqueeze(-1), 1.0)
        return type("Output", (), {"logits": logits})


def test_gold_answer_mask_matches_prompt_and_two_tokens_per_packet():
    tokenizer = FakeTokenizer()
    ids, labels = build_gold_answer_inputs(tokenizer, 99, "Question?", "answer", 3)
    assert int((ids == 99).sum()) == 6
    first_label = int((labels != -100).nonzero()[0])
    assert torch.all(labels[:first_label] == -100)
    assert labels[-1] == tokenizer.eos_token_id
    assert int((labels != -100).sum()) >= 2


def test_mean_nll_is_answer_token_normalized_and_delta_sign_is_correct():
    logits = torch.zeros(2, 5, 4)
    labels = torch.tensor([[-100, -100, 1, 1, 1], [-100, -100, -100, 1, 1]])
    logits[0, 1:4, 1] = 2.0
    logits[1, 2:4, 1] = 2.0
    nll = mean_answer_nll_from_logits(logits, labels)
    assert nll[0].item() == pytest.approx(nll[1].item(), abs=1e-7)
    assert delta_utility(1.5, 1.0) == pytest.approx(0.5)
    assert delta_utility(1.0, 1.5) == pytest.approx(-0.5)


def test_static_rollout_stop_and_fixed_budget_rules():
    utilities = {1: .3, 2: .1, 3: -.2, 4: .05}
    selected, _ = static_utility_rollout(utilities)
    assert selected == [1, 2, 4]
    fixed, _ = static_utility_rollout(utilities, fixed_k=2)
    assert fixed == [1, 2]
    forced, actions = static_utility_rollout({8: -.1, 9: -.2})
    assert forced == [8] and actions[0]["delta_utility"] == -.1


def test_state_rollout_stop_and_max_packet_rule():
    assert choose_state_utility_action({1: -.1, 2: -.2}, selected_count=0) == (1, -.1)
    assert choose_state_utility_action({1: -.1, 2: -.2}, selected_count=1) == (None, -.1)
    utilities = {index: 1.0 - index / 100 for index in range(20)}
    selected, _ = static_utility_rollout(utilities, max_packets=6)
    assert len(selected) == 6


def test_paired_bootstrap_requires_alignment_and_is_reproducible():
    left = [{"sample_id": str(i), "short_f1": 1.0, "num_packets": 1} for i in range(5)]
    right = [{"sample_id": str(i), "short_f1": 0.0, "num_packets": 2} for i in range(5)]
    first = paired_bootstrap(left, right, samples=100, seed=42)
    second = paired_bootstrap(left, right, samples=100, seed=42)
    assert first == second
    assert first["short_f1_delta"] == 100.0
    assert first["avg_packet_delta"] == -1.0
    with pytest.raises(ValueError, match="alignment"):
        paired_bootstrap(left, right[:-1], samples=10)


def test_cache_key_is_order_invariant_and_unique_by_candidate():
    assert utility_cache_key("s", [4, 1], 3) == utility_cache_key("s", [1, 4], 3)
    assert utility_cache_key("s", [1, 4], 3) != utility_cache_key("s", [1, 4], 2)


def test_independent_same_shape_nll_recompute_is_exact():
    tokenizer = FakeTokenizer(); model = FakeGenerator().eval()
    embeddings = torch.zeros(3, 8)
    groups = [[], [0], [1]]
    first = gold_answer_nll_batch(
        model, tokenizer, 99, "q", "a", embeddings, groups, torch.device("cpu")
    )
    second = gold_answer_nll_batch(
        model, tokenizer, 99, "q", "a", embeddings, groups, torch.device("cpu")
    )
    assert first == second
