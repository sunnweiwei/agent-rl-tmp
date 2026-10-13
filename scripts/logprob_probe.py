#!/usr/bin/env python3
"""Compare vLLM sampled-token logprobs with HF forward logprobs.

This is a diagnostic helper for rollout-correction debugging. It intentionally
works in two phases so vLLM and HF do not need to hold GPU memory at the same
time:

  1. generate: save prompt ids, generated ids, and vLLM logprobs to JSON.
  2. compare: reload the JSON and compute HF logprobs for the same ids.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from collections.abc import Mapping
from typing import Any


def _token_ids(x: Any) -> list[int]:
    if isinstance(x, Mapping):
        x = x["input_ids"]
    if hasattr(x, "tolist"):
        x = x.tolist()
    if x and isinstance(x[0], list):
        assert len(x) == 1
        x = x[0]
    return [int(t) for t in x]


def _load_messages(data_file: str, index: int) -> list[dict[str, Any]]:
    import pandas as pd

    row = pd.read_parquet(data_file).iloc[index]
    messages = row["prompt"]
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    return [dict(m) for m in messages]


def _apply_chat_template(tokenizer, messages: list[dict[str, Any]], enable_thinking: bool | None) -> tuple[str, list[int], str]:
    kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        ids = _token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kwargs))
        used = f"enable_thinking={enable_thinking}"
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = _token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        used = "enable_thinking=unsupported"
    return text, ids, used


def _roundtrip_info(tokenizer, text: str, ids: list[int]) -> dict[str, Any]:
    ids_from_text = _token_ids(tokenizer(text, add_special_tokens=False, return_attention_mask=False))
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    ids_from_decoded = _token_ids(tokenizer(decoded, add_special_tokens=False, return_attention_mask=False))
    first_text_mismatch = None
    for i, (a, b) in enumerate(zip(ids, ids_from_text, strict=False)):
        if a != b:
            first_text_mismatch = i
            break
    first_decoded_mismatch = None
    for i, (a, b) in enumerate(zip(ids, ids_from_decoded, strict=False)):
        if a != b:
            first_decoded_mismatch = i
            break
    return {
        "ids_len": len(ids),
        "ids_from_text_len": len(ids_from_text),
        "ids_from_decoded_len": len(ids_from_decoded),
        "text_roundtrip_equal": ids == ids_from_text,
        "decoded_roundtrip_equal": ids == ids_from_decoded,
        "first_text_mismatch": first_text_mismatch,
        "first_decoded_mismatch": first_decoded_mismatch,
    }


def generate(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    samples = []
    prompts = []
    template_used = None
    for data_index in range(args.data_index, args.data_index + args.num_prompts):
        messages = _load_messages(args.data_file, data_index)
        prompt_text, prompt_ids, template_used = _apply_chat_template(tokenizer, messages, args.enable_thinking)
        samples.append(
            {
                "data_index": data_index,
                "messages": messages,
                "prompt_text": prompt_text,
                "prompt_ids": prompt_ids,
                "prompt_roundtrip": _roundtrip_info(tokenizer, prompt_text, prompt_ids),
            }
        )
        prompts.append({"prompt_token_ids": prompt_ids})

    llm_kwargs = dict(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
        disable_custom_all_reduce=True,
        logprobs_mode=args.logprobs_mode,
    )
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        logprobs=0,
    )
    outputs = llm.generate(prompts, sampling_params)
    outputs = sorted(outputs, key=lambda o: int(o.request_id))
    for sample, output in zip(samples, outputs, strict=True):
        out = output.outputs[0]
        response_ids = [int(t) for t in out.token_ids]
        response_logprobs = [float(lp[response_ids[i]].logprob) for i, lp in enumerate(out.logprobs)]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
        response_ids_from_text = _token_ids(tokenizer(response_text, add_special_tokens=False, return_attention_mask=False))
        sample.update(
            {
                "response_text": response_text,
                "response_ids": response_ids,
                "vllm_logprobs": response_logprobs,
                "response_roundtrip": {
                    "ids_len": len(response_ids),
                    "ids_from_text_len": len(response_ids_from_text),
                    "text_roundtrip_equal": response_ids == response_ids_from_text,
                    "first_text_mismatch": next(
                        (
                            i
                            for i, (a, b) in enumerate(zip(response_ids, response_ids_from_text, strict=False))
                            if a != b
                        ),
                        None,
                    ),
                },
            }
        )

    payload = {
        "model": args.model,
        "data_file": args.data_file,
        "data_index": args.data_index,
        "template_used": template_used,
        "logprobs_mode": args.logprobs_mode,
        "sampling": {
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
        },
        "samples": samples,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "model": payload["model"],
                "logprobs_mode": payload["logprobs_mode"],
                "template_used": payload["template_used"],
                "num_samples": len(samples),
                "prompt_roundtrip_all": all(s["prompt_roundtrip"]["text_roundtrip_equal"] for s in samples),
                "response_roundtrip_all": all(s["response_roundtrip"]["text_roundtrip_equal"] for s in samples),
                "response_lens": [s["response_roundtrip"]["ids_len"] for s in samples],
            },
            indent=2,
        )
    )
    print(f"saved {args.output}")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, round((len(xs) - 1) * p)))
    return xs[idx]


def compare(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    payload = json.loads(Path(args.input).read_text())
    model_path = args.model or payload["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    if "samples" not in payload:
        payload["samples"] = [
            {
                "data_index": payload.get("data_index", payload.get("data_index", 0)),
                "prompt_ids": payload["prompt_ids"],
                "response_ids": payload["response_ids"],
                "vllm_logprobs": payload["vllm_logprobs"],
                "prompt_roundtrip": payload.get("prompt_roundtrip"),
                "response_roundtrip": payload.get("response_roundtrip"),
            }
        ]

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, args.torch_dtype),
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map if args.device_map else None,
    )
    if not args.device_map:
        model.to(args.device)
    model.eval()

    sample_summaries = []
    all_rows = []
    for sample_idx, sample in enumerate(payload["samples"]):
        prompt_ids = [int(t) for t in sample["prompt_ids"]]
        response_ids = [int(t) for t in sample["response_ids"]]
        vllm_logprobs = [float(x) for x in sample["vllm_logprobs"]]
        assert len(response_ids) == len(vllm_logprobs), (len(response_ids), len(vllm_logprobs))

        ids = torch.tensor(
            [prompt_ids + response_ids],
            dtype=torch.long,
            device=args.device if not args.device_map else model.device,
        )
        attention_mask = torch.ones_like(ids)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attention_mask, use_cache=False)
            logits = out.logits[:, len(prompt_ids) - 1 : len(prompt_ids) - 1 + len(response_ids), :]
            labels = ids[:, len(prompt_ids) : len(prompt_ids) + len(response_ids)]
            hf_logprobs = torch.gather(F.log_softmax(logits.float(), dim=-1), -1, labels.unsqueeze(-1)).squeeze(-1)
        hf = [float(x) for x in hf_logprobs[0].detach().cpu()]
        sample_diffs = [h - v for h, v in zip(hf, vllm_logprobs, strict=True)]
        sample_abs = [abs(x) for x in sample_diffs]
        sample_summaries.append(
            {
                "sample_idx": sample_idx,
                "data_index": sample.get("data_index"),
                "prompt_len": len(prompt_ids),
                "num_response_tokens": len(response_ids),
                "mean_hf_minus_vllm": sum(sample_diffs) / len(sample_diffs),
                "mean_abs_diff": sum(sample_abs) / len(sample_abs),
                "p99_abs_diff": _percentile(sample_abs, 0.99),
                "max_abs_diff": max(sample_abs),
                "prompt_roundtrip": sample.get("prompt_roundtrip"),
                "response_roundtrip": sample.get("response_roundtrip"),
            }
        )
        for pos, (token_id, h, v, d) in enumerate(zip(response_ids, hf, vllm_logprobs, sample_diffs, strict=True)):
            all_rows.append((sample_idx, sample.get("data_index"), pos, token_id, h, v, d))

    diffs = [r[6] for r in all_rows]
    abs_diffs = [abs(x) for x in diffs]
    top = sorted(range(len(abs_diffs)), key=lambda i: abs_diffs[i], reverse=True)[: args.topn]
    buckets = []
    bucket_size = args.bucket_size
    for start in range(0, len(diffs), bucket_size):
        chunk = diffs[start : start + bucket_size]
        if not chunk:
            continue
        buckets.append(
            {
                "start": start,
                "end": start + len(chunk) - 1,
                "mean_hf_minus_vllm": sum(chunk) / len(chunk),
                "mean_abs": sum(abs(x) for x in chunk) / len(chunk),
                "max_abs": max(abs(x) for x in chunk),
            }
        )

    summary = {
        "model": model_path,
        "logprobs_mode": payload.get("logprobs_mode"),
        "num_samples": len(payload["samples"]),
        "num_response_tokens": len(all_rows),
        "mean_hf_minus_vllm": sum(diffs) / len(diffs),
        "mean_abs_diff": sum(abs_diffs) / len(abs_diffs),
        "p50_abs_diff": _percentile(abs_diffs, 0.50),
        "p90_abs_diff": _percentile(abs_diffs, 0.90),
        "p99_abs_diff": _percentile(abs_diffs, 0.99),
        "max_abs_diff": max(abs_diffs),
        "top": [
            {
                "sample_idx": all_rows[i][0],
                "data_index": all_rows[i][1],
                "pos": all_rows[i][2],
                "token_id": all_rows[i][3],
                "token": tokenizer.decode([all_rows[i][3]], skip_special_tokens=False),
                "hf_logprob": all_rows[i][4],
                "vllm_logprob": all_rows[i][5],
                "hf_minus_vllm": all_rows[i][6],
                "abs_diff": abs(all_rows[i][6]),
            }
            for i in top
        ],
        "buckets": buckets,
        "sample_summaries": sample_summaries,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"saved {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--model", required=True)
    gen.add_argument("--data-file", required=True)
    gen.add_argument("--data-index", type=int, default=0)
    gen.add_argument("--num-prompts", type=int, default=1)
    gen.add_argument("--output", required=True)
    gen.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    gen.add_argument("--logprobs-mode", default="raw_logprobs", choices=["raw_logprobs", "processed_logprobs", "raw_logits", "processed_logits"])
    gen.add_argument("--tensor-parallel-size", type=int, default=1)
    gen.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    gen.add_argument("--max-model-len", type=int, default=2048)
    gen.add_argument("--max-tokens", type=int, default=256)
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--top-k", type=int, default=-1)
    gen.add_argument("--repetition-penalty", type=float, default=1.0)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--dtype", default="bfloat16")
    gen.add_argument("--enforce-eager", action="store_true")
    gen.add_argument("--trust-remote-code", action="store_true")
    gen.set_defaults(func=generate)

    cmp = sub.add_parser("compare")
    cmp.add_argument("--input", required=True)
    cmp.add_argument("--output")
    cmp.add_argument("--model")
    cmp.add_argument("--device", default="cuda:0")
    cmp.add_argument("--device-map", default="")
    cmp.add_argument("--torch-dtype", default="bfloat16")
    cmp.add_argument("--attn-implementation", default="flash_attention_2")
    cmp.add_argument("--trust-remote-code", action="store_true")
    cmp.add_argument("--topn", type=int, default=20)
    cmp.add_argument("--bucket-size", type=int, default=64)
    cmp.set_defaults(func=compare)

    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.func(args)


if __name__ == "__main__":
    main()
