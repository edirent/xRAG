import types

import torch


def prepare_multi_token_inputs_embeds(model, input_ids, retrieval_embeds):
    """Project packets, flatten packet-major, and replace consecutive XRAG slots."""
    inputs_embeds = model.model.embed_tokens(input_ids)
    retrieval_embeds = retrieval_embeds.reshape(-1, model.retriever_hidden_size)
    num_packets = retrieval_embeds.shape[0]
    projected = model.projector(retrieval_embeds.to(inputs_embeds.dtype))
    expected_shape = (num_packets, model.projector.tokens_per_packet, inputs_embeds.shape[-1])
    assert projected.shape == expected_shape, (projected.shape, expected_shape)
    flattened = projected.reshape(num_packets * model.projector.tokens_per_packet, inputs_embeds.shape[-1])
    num_xrag_tokens = int(torch.sum(input_ids == model.xrag_token_id))
    assert num_xrag_tokens == flattened.shape[0], (num_xrag_tokens, flattened.shape[0])
    inputs_embeds[input_ids == model.xrag_token_id] = flattened
    return inputs_embeds


def install_multi_token_injection(model):
    def prepare_inputs_embeds(self, input_ids, retrieval_embeds):
        return prepare_multi_token_inputs_embeds(self, input_ids, retrieval_embeds)
    model.prepare_inputs_embeds = types.MethodType(prepare_inputs_embeds, model)
    return model
