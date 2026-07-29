#!/usr/bin/env python
import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.language_modeling.utils import XRAG_TOKEN
from src.model import SFR, XMistralForCausalLM


SFR_MODEL_NAME = "Salesforce/SFR-Embedding-Mistral"
XRAG_MODEL_NAME = "Hannibal046/xrag-7b"


def vram_gb(fn):
    return fn() / 1024**3


def print_vram(label):
    print(f"{label} allocated GB: {vram_gb(torch.cuda.memory_allocated):.3f}")
    print(f"{label} reserved GB: {vram_gb(torch.cuda.memory_reserved):.3f}")


def load_models(device, dtype):
    print(f"Device: {device}")
    print(f"Requested dtype: {dtype}")

    sfr_tokenizer = AutoTokenizer.from_pretrained(SFR_MODEL_NAME)
    sfr_model = SFR.from_pretrained(SFR_MODEL_NAME, torch_dtype=dtype).eval().to(device)
    print_vram("After SFR")
    print("SFR device:", next(sfr_model.parameters()).device)
    print("SFR dtype:", next(sfr_model.parameters()).dtype)
    print("SFR embedding dim:", sfr_model.get_embed_dim())
    print("SFR embedding length:", sfr_model.get_embed_length())

    xrag_tokenizer = AutoTokenizer.from_pretrained(XRAG_MODEL_NAME)
    config = AutoConfig.from_pretrained(XRAG_MODEL_NAME)
    xrag_model = XMistralForCausalLM.from_pretrained(
        XRAG_MODEL_NAME,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval().to(device)

    assert XRAG_TOKEN in xrag_tokenizer.get_vocab(), f"{XRAG_TOKEN} missing from tokenizer"
    xrag_token_id = xrag_tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
    xrag_model.set_xrag_token_id(xrag_token_id)

    print_vram("After both models")
    print("xRAG device:", next(xrag_model.parameters()).device)
    print("xRAG dtype:", next(xrag_model.parameters()).dtype)
    print("xRAG token:", XRAG_TOKEN)
    print("xRAG token id:", xrag_token_id)

    assert next(sfr_model.parameters()).device.type == "cuda"
    assert next(xrag_model.parameters()).device.type == "cuda"
    assert next(sfr_model.parameters()).device.index == 0
    assert next(xrag_model.parameters()).device.index == 0

    return sfr_tokenizer, sfr_model, xrag_tokenizer, xrag_model, xrag_token_id


@torch.no_grad()
def encode_documents(tokenizer, model, documents, device):
    start = time.perf_counter()
    tokenized = tokenizer(
        documents,
        max_length=180,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    retrieval_embeds = model.get_doc_embedding(
        input_ids=tokenized.input_ids,
        attention_mask=tokenized.attention_mask,
    )
    retrieval_embeds = retrieval_embeds.view(-1, retrieval_embeds.shape[-1])
    encode_seconds = time.perf_counter() - start
    return retrieval_embeds, encode_seconds


def build_prompt(question, num_retrieval_embeddings):
    background = " ".join([XRAG_TOKEN] * num_retrieval_embeddings)
    return (
        "Refer to the background document and answer the question.\n"
        f"Background: {background}\n"
        f"Question: {question}\n"
        "Answer:"
    )


@torch.no_grad()
def run_case(name, documents, question, expected_answer, ctx):
    sfr_tokenizer, sfr_model, xrag_tokenizer, xrag_model, xrag_token_id, device = ctx

    print()
    print("=" * 80)
    print(f"Case: {name}")
    print("Question:", question)
    print("Expected answer:", expected_answer)

    retrieval_embeds, encode_seconds = encode_documents(
        sfr_tokenizer,
        sfr_model,
        documents,
        device,
    )

    print("Retrieval embedding shape:", tuple(retrieval_embeds.shape))
    print(f"SFR encode time seconds: {encode_seconds:.3f}")
    assert retrieval_embeds.ndim == 2
    assert retrieval_embeds.shape[0] == len(documents)
    assert retrieval_embeds.shape[-1] == 4096

    prompt = build_prompt(question, retrieval_embeds.shape[0])
    tokenized_prompt = xrag_tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = tokenized_prompt.input_ids
    attention_mask = tokenized_prompt.attention_mask

    num_xrag_tokens = (input_ids == xrag_token_id).sum().item()
    num_retrieval_embeddings = retrieval_embeds.shape[0]
    print("XRAG tokens:", num_xrag_tokens)
    print("Retrieval embeddings:", num_retrieval_embeddings)
    assert num_xrag_tokens == num_retrieval_embeddings

    start = time.perf_counter()
    generated_output = xrag_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        retrieval_embeds=retrieval_embeds,
        do_sample=False,
        max_new_tokens=32,
        pad_token_id=xrag_tokenizer.eos_token_id,
    )
    generate_seconds = time.perf_counter() - start

    generated_text = xrag_tokenizer.batch_decode(
        generated_output,
        skip_special_tokens=True,
    )[0]
    generated_tokens = generated_output.shape[-1]
    print("Generated text:", generated_text.strip())
    print(f"xRAG generation time seconds: {generate_seconds:.3f}")
    print("Number of retrieval embeddings:", num_retrieval_embeddings)
    print("Number of generated tokens:", generated_tokens)

    assert generated_text.strip()
    return generated_text


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 0 xRAG dual-model smoke test.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-multi", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available()
    assert torch.cuda.is_bf16_supported()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    ctx = (*load_models(device, dtype), device)

    run_case(
        name="single embedding",
        documents=["Paris is the capital and most populous city of France."],
        question="What is the capital of France?",
        expected_answer="Paris",
        ctx=ctx,
    )

    if not args.skip_multi:
        run_case(
            name="three embeddings",
            documents=[
                "Paris is the capital of France.",
                "Berlin is the capital of Germany.",
                "Madrid is the capital of Spain.",
            ],
            question="What is the capital of Germany?",
            expected_answer="Berlin",
            ctx=ctx,
        )

    print()
    print("Peak allocated GB:", vram_gb(torch.cuda.max_memory_allocated))
    print("Peak reserved GB:", vram_gb(torch.cuda.max_memory_reserved))
    print("Smoke test completed.")


if __name__ == "__main__":
    main()
