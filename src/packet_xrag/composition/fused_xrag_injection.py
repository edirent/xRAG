"""Inject already-fused LLM-space tokens without applying the packet projector again."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from scripts.packet_xrag import train_packet_projector as v1


def build_fused_answer_inputs(tokenizer, xrag_id, question, answer, output_slots=4):
    prompt = v1.build_prompt(question, output_slots)
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt + " " + answer + tokenizer.eos_token,
                         add_special_tokens=False).input_ids
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    if sum(token == xrag_id for token in full_ids) != output_slots:
        raise RuntimeError("fused prompt placeholder count mismatch")
    return torch.tensor(full_ids), torch.tensor(labels)


def pad_fused_answer_batch(tokenizer, items, device):
    input_ids = pad_sequence([item[0] for item in items], batch_first=True,
                             padding_value=tokenizer.pad_token_id, padding_side="left").to(device)
    labels = pad_sequence([item[1] for item in items], batch_first=True,
                          padding_value=-100, padding_side="left").to(device)
    return input_ids, labels, input_ids.ne(tokenizer.pad_token_id)


def replace_xrag_placeholders(model, input_ids, xrag_id, fused_tokens):
    if fused_tokens.ndim != 3 or fused_tokens.shape[0] != input_ids.shape[0]:
        raise ValueError("fused tokens must be [B,M,D]")
    counts = input_ids.eq(xrag_id).sum(dim=1)
    if not bool(counts.eq(fused_tokens.shape[1]).all()):
        raise RuntimeError("each prompt must contain exactly M XRAG placeholders")
    embeddings = model.model.embed_tokens(input_ids)
    embeddings = embeddings.clone()
    embeddings[input_ids == xrag_id] = fused_tokens.reshape(-1, fused_tokens.shape[-1]).to(
        embeddings.dtype)
    return embeddings


def fused_answer_loss(model, input_ids, attention_mask, labels, xrag_id, fused_tokens):
    embeddings = replace_xrag_placeholders(model, input_ids, xrag_id, fused_tokens)
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask,
                    use_cache=False)
    logits = outputs.logits[:, :-1].float(); targets = labels[:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           ignore_index=-100)


@torch.inference_mode()
def greedy_generate_fused(model, tokenizer, input_ids, attention_mask, xrag_id,
                          fused_tokens, max_new_tokens=32):
    embeddings = replace_xrag_placeholders(model, input_ids, xrag_id, fused_tokens)
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask,
                    use_cache=True)
    past = outputs.past_key_values; next_token = outputs.logits[:, -1].argmax(-1, keepdim=True)
    generated = [next_token]; finished = next_token.squeeze(-1).eq(tokenizer.eos_token_id)
    current_mask = attention_mask
    for _ in range(1, max_new_tokens):
        current_mask = torch.cat([current_mask, torch.ones(current_mask.shape[0], 1,
                                                           device=current_mask.device,
                                                           dtype=current_mask.dtype)], 1)
        outputs = model(input_ids=next_token, attention_mask=current_mask,
                        past_key_values=past, use_cache=True)
        past = outputs.past_key_values; next_token = outputs.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(next_token); finished |= next_token.squeeze(-1).eq(tokenizer.eos_token_id)
        if bool(finished.all()): break
    return torch.cat(generated, 1)

