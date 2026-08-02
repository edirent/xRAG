import types

import torch


def prepare_token_state_inputs_embeds(
    model, input_ids, token_states, token_mask, pooled_embeddings
):
    inputs_embeds = model.model.embed_tokens(input_ids)
    projected = model.projector(
        token_states.to(inputs_embeds.dtype),
        token_mask,
        pooled_embeddings.to(inputs_embeds.dtype),
    )
    expected = (token_states.shape[0], 2, inputs_embeds.shape[-1])
    if projected.shape != expected:
        raise ValueError(f"projected shape {projected.shape} != {expected}")
    flattened = projected.reshape(-1, inputs_embeds.shape[-1])
    slots = input_ids == model.xrag_token_id
    if int(slots.sum()) != flattened.shape[0]:
        raise ValueError("XRAG slot count does not match two tokens per packet")
    inputs_embeds[slots] = flattened
    return inputs_embeds


def install_token_state_injection(model):
    def prepare_inputs_embeds(self, input_ids, retrieval_embeds):
        if not isinstance(retrieval_embeds, dict):
            raise TypeError("token-state retrieval_embeds must be a mapping")
        return prepare_token_state_inputs_embeds(
            self,
            input_ids,
            retrieval_embeds["token_states"],
            retrieval_embeds["token_mask"],
            retrieval_embeds["pooled_embeddings"],
        )

    model.prepare_inputs_embeds = types.MethodType(prepare_inputs_embeds, model)
    return model
