#!/usr/bin/env python3
"""Check Qwen MTP correctness while vLLM prefix caching is enabled."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def _prompt_ids(tokenizer: Any, answer: int, target_len: int) -> list[int]:
    filler = (
        "Reference ledger entry: the warehouse audit is complete and all ordinary "
        "rows are informational. Continue reading until the final control record.\n"
    )
    suffix = (
        f"\nFinal control record: answer_code={answer}. "
        "Return only the integer answer_code and nothing else."
    )

    def render(repetitions: int) -> list[int]:
        messages = [{"role": "user", "content": filler * repetitions + suffix}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if hasattr(rendered, "input_ids"):
            rendered = rendered.input_ids
        if isinstance(rendered, dict):
            rendered = rendered["input_ids"]
        if rendered and isinstance(rendered[0], list):
            rendered = rendered[0]
        return list(rendered)

    filler_tokens = max(1, len(tokenizer.encode(filler, add_special_tokens=False)))
    base = render(0)
    repetitions = max(0, (target_len - len(base)) // filler_tokens)
    ids = render(repetitions)
    while len(ids) > target_len and repetitions > 0:
        repetitions -= 1
        ids = render(repetitions)
    return ids


def _extract(output: Any, tokenizer: Any, expected_answer: int) -> dict[str, Any]:
    sequence = output.outputs[0]
    token_ids = list(sequence.token_ids)
    logprobs: list[float] = []
    if sequence.logprobs is not None:
        for token_id, token_logprobs in zip(token_ids, sequence.logprobs, strict=True):
            logprobs.append(float(token_logprobs[token_id].logprob))

    metrics = getattr(output, "metrics", None)
    stats = getattr(metrics, "request_spec_decode_stats", None)
    return {
        "token_ids": token_ids,
        "logprobs": logprobs,
        "num_cached_tokens": int(getattr(output, "num_cached_tokens", 0) or 0),
        "spec_num_draft_tokens": int(getattr(stats, "num_draft_tokens", 0) or 0),
        "spec_num_accepted_tokens": int(getattr(stats, "num_accepted_tokens", 0) or 0),
        "spec_num_verify_steps": int(getattr(stats, "num_verify_steps", 0) or 0),
        "text": tokenizer.decode(token_ids, skip_special_tokens=True),
        "expected_answer": expected_answer,
    }


def _run_pass(
    llm: LLM,
    tokenizer: Any,
    prompts: list[dict[str, list[int]]],
    expected_answers: list[int],
    max_tokens: int,
) -> tuple[float, list[dict[str, Any]]]:
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=False,
        detokenize=True,
        logprobs=1,
    )
    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
    return time.perf_counter() - started, [
        _extract(output, tokenizer, answer)
        for output, answer in zip(outputs, expected_answers, strict=True)
    ]


def _compare_passes(reference: list[dict[str, Any]], actual: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = list(zip(reference, actual, strict=True))
    exact_sequences = sum(left["token_ids"] == right["token_ids"] for left, right in pairs)
    positional_matches = 0
    positional_tokens = 0
    common_prefix_lengths = []
    diffs = []
    for left, right in pairs:
        left_ids = left["token_ids"]
        right_ids = right["token_ids"]
        positional_tokens += max(len(left_ids), len(right_ids))
        positional_matches += sum(a == b for a, b in zip(left_ids, right_ids))
        common_prefix = 0
        for left_id, right_id in zip(left_ids, right_ids):
            if left_id != right_id:
                break
            common_prefix += 1
        common_prefix_lengths.append(common_prefix)
        diffs.extend(
            abs(left_value - right_value)
            for left_value, right_value in zip(
                left["logprobs"][:common_prefix],
                right["logprobs"][:common_prefix],
            )
        )
    return {
        "token_ids_match": exact_sequences == len(pairs),
        "exact_sequence_rate": exact_sequences / len(pairs),
        "positional_token_match_rate": positional_matches / positional_tokens if positional_tokens else 1.0,
        "min_common_prefix_tokens": min(common_prefix_lengths, default=0),
        "max_abs_logprob_diff": max(diffs, default=0.0),
        "mean_abs_logprob_diff": sum(diffs) / len(diffs) if diffs else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", choices=("no_mtp", "mtp1"), required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--reference-json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=8192)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    expected_answers = [1000 + index for index in range(args.batch_size)]
    mixed_lengths = ([128, 512, 2048] + [args.input_len] * args.batch_size)[: args.batch_size]
    prompt_ids = [
        _prompt_ids(tokenizer, answer, target_len)
        for answer, target_len in zip(expected_answers, mixed_lengths, strict=True)
    ]
    prompts = [{"prompt_token_ids": ids} for ids in prompt_ids]

    speculative_config = None
    if args.variant == "mtp1":
        speculative_config = {
            "method": "mtp",
            "num_speculative_tokens": 1,
            "attention_backend": "TRITON_ATTN",
        }
    speculative_config_record = dict(speculative_config) if speculative_config is not None else None

    load_started = time.perf_counter()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max(map(len, prompt_ids)) + args.output_len + 128,
        max_num_batched_tokens=max(65536, sum(map(len, prompt_ids))),
        max_num_seqs=max(32, args.batch_size),
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        generation_config="vllm",
        load_format="auto",
        disable_log_stats=False,
        speculative_config=dict(speculative_config_record) if speculative_config_record is not None else None,
        attention_backend="TRITON_ATTN",
        gdn_prefill_backend="triton",
    )
    load_seconds = time.perf_counter() - load_started

    cold_seconds, cold = _run_pass(llm, tokenizer, prompts, expected_answers, args.output_len)
    warm_seconds, warm = _run_pass(llm, tokenizer, prompts, expected_answers, args.output_len)
    repeated = _compare_passes(cold, warm)

    reset_ok = bool(llm.llm_engine.reset_prefix_cache())
    reset_seconds, after_reset = _run_pass(llm, tokenizer, prompts, expected_answers, args.output_len)
    reset_comparison = _compare_passes(cold, after_reset)

    query_tokens = sum(map(len, prompt_ids))
    warm_cached_tokens = sum(item["num_cached_tokens"] for item in warm)
    draft_tokens = sum(item["spec_num_draft_tokens"] for item in warm)
    accepted_tokens = sum(item["spec_num_accepted_tokens"] for item in warm)
    result = {
        "variant": args.variant,
        "speculative_config": speculative_config_record,
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "actual_input_lengths": list(map(len, prompt_ids)),
        "output_len": args.output_len,
        "load_seconds": load_seconds,
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "after_reset_seconds": reset_seconds,
        "warm_prefix_cache_hit_rate": warm_cached_tokens / query_tokens,
        "warm_cached_tokens": warm_cached_tokens,
        "prefix_cache_query_tokens": query_tokens,
        "spec_acceptance_rate": accepted_tokens / draft_tokens if draft_tokens else None,
        "spec_num_draft_tokens": draft_tokens,
        "spec_num_accepted_tokens": accepted_tokens,
        "reset_prefix_cache_ok": reset_ok,
        "repeat_comparison": repeated,
        "after_reset_comparison": reset_comparison,
        "cold": cold,
        "warm": warm,
        "after_reset": after_reset,
        "cold_answer_accuracy": sum(str(item["expected_answer"]) in item["text"] for item in cold) / len(cold),
        "warm_answer_accuracy": sum(str(item["expected_answer"]) in item["text"] for item in warm) / len(warm),
    }

    reference_comparison = None
    if args.reference_json:
        reference = json.loads(Path(args.reference_json).read_text())
        reference_comparison = _compare_passes(reference["warm"], warm)
        result["reference_variant"] = reference["variant"]
        result["reference_comparison"] = reference_comparison

    max_logprob_diff = max(
        reset_comparison["max_abs_logprob_diff"],
        reference_comparison["max_abs_logprob_diff"] if reference_comparison else 0.0,
    )
    result["passed"] = bool(
        reset_comparison["token_ids_match"]
        and max_logprob_diff <= 5e-3
        and result["warm_prefix_cache_hit_rate"] >= 0.90
        and reset_ok
        and math.isfinite(max_logprob_diff)
        and result["cold_answer_accuracy"] == 1.0
        and result["warm_answer_accuracy"] == 1.0
        and (reference_comparison is None or reference_comparison["exact_sequence_rate"] >= 0.80)
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"cold", "warm", "after_reset"}}, indent=2))

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
