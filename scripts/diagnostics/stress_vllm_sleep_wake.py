#!/usr/bin/env python3
"""Stress vLLM's staged sleep/wake path with deterministic long prompts."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request


def request_json(
    base_url: str,
    path: str,
    payload: dict | None = None,
    query: dict | None = None,
    timeout: float = 300.0,
) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query, doseq=True)}"
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body else {}


def post_empty(
    base_url: str,
    path: str,
    query: dict | None = None,
    timeout: float = 300.0,
) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query, doseq=True)}"
    request = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body else {}


def post_payload(
    base_url: str,
    path: str,
    payload: dict,
    timeout: float = 300.0,
) -> None:
    url = f"{base_url.rstrip('/')}{path}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


def reset_prefix_cache(base_url: str) -> None:
    try:
        post_empty(base_url, "/reset_prefix_cache", timeout=30.0)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise


def make_prompts(count: int, approximate_tokens: int) -> list[str]:
    common_line = (
        "A data analyst checks a database query, validates every assumption, "
        "and returns only the final numeric result. "
    )
    common_prefix = common_line * max(1, approximate_tokens // 20)
    return [
        f"{common_prefix}\nCase {index}: What is {index} plus {index + 7}?"
        for index in range(count)
    ]


def make_token_prompts(
    count: int,
    prompt_tokens: int,
    common_prefix_tokens: int,
) -> list[list[int]]:
    if common_prefix_tokens >= prompt_tokens:
        raise ValueError("common prefix must be shorter than each prompt")
    common = ([14556, 1814] * ((common_prefix_tokens + 1) // 2))[
        :common_prefix_tokens
    ]
    prompts = []
    for index in range(count):
        # Vary lengths around the target like a real agent batch while retaining
        # an exactly shared cacheable prefix.
        length = prompt_tokens + (index % 9 - 4) * 37
        suffix_length = length - common_prefix_tokens
        suffix = [1000 + ((index * 7919 + offset * 104729) % 200000) for offset in range(suffix_length)]
        prompts.append(common + suffix)
    return prompts


def generate_one(
    base_url: str,
    model: str,
    prompt: str | list[int],
    max_tokens: int,
) -> str:
    response = request_json(
        base_url,
        "/v1/completions",
        payload={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 1234,
        },
    )
    return response["choices"][0]["text"]


def generate_batch(
    base_url: str,
    model: str,
    prompts: list[str | list[int]],
    max_tokens: int,
) -> list[str]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [
            pool.submit(generate_one, base_url, model, prompt, max_tokens)
            for prompt in prompts
        ]
        return [future.result() for future in futures]


def digest(outputs: list[str]) -> str:
    encoded = json.dumps(outputs, ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="qwen36-27b")
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int, default=3072)
    parser.add_argument("--exact-token-prompts", action="store_true")
    parser.add_argument("--common-prefix-tokens", type=int, default=3136)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--lora-name")
    parser.add_argument("--lora-path")
    parser.add_argument("--tensor-lora-path")
    parser.add_argument("--tensor-lora-id", type=int, default=1)
    parser.add_argument("--tensor-lora-name", default="synthetic")
    args = parser.parse_args()
    if bool(args.lora_name) != bool(args.lora_path):
        parser.error("--lora-name and --lora-path must be provided together")
    if args.lora_path and args.tensor_lora_path:
        parser.error("file and tensor LoRA reload modes are mutually exclusive")

    if args.exact_token_prompts:
        prompts = make_token_prompts(
            args.concurrency,
            args.prompt_tokens,
            args.common_prefix_tokens,
        )
    else:
        prompts = make_prompts(args.concurrency, args.prompt_tokens)
    request_json(args.base_url, "/health")

    started = time.monotonic()
    baseline = generate_batch(args.base_url, args.model, prompts, args.max_tokens)
    baseline_digest = digest(baseline)
    print(
        json.dumps(
            {
                "event": "baseline",
                "digest": baseline_digest,
                "seconds": round(time.monotonic() - started, 3),
            }
        ),
        flush=True,
    )

    for cycle in range(1, args.cycles + 1):
        cycle_started = time.monotonic()
        post_empty(args.base_url, "/sleep", query={"level": 1})
        post_empty(args.base_url, "/wake_up", query={"tags": "weights"})
        if args.lora_name:
            post_payload(
                args.base_url,
                "/v1/unload_lora_adapter",
                {"lora_name": args.lora_name},
            )
            post_payload(
                args.base_url,
                "/v1/load_lora_adapter",
                {"lora_name": args.lora_name, "lora_path": args.lora_path},
            )
        elif args.tensor_lora_path:
            request_json(
                args.base_url,
                "/collective_rpc",
                payload={
                    "method": "reload_tensor_lora",
                    "args": [
                        json.dumps(
                            {
                                "adapter_path": args.tensor_lora_path,
                                "lora_int_id": args.tensor_lora_id,
                                "lora_name": args.tensor_lora_name,
                            }
                        )
                    ],
                    "timeout": 300,
                },
            )
        reset_prefix_cache(args.base_url)
        post_empty(args.base_url, "/wake_up", query={"tags": "kv_cache"})
        reset_prefix_cache(args.base_url)
        outputs = generate_batch(args.base_url, args.model, prompts, args.max_tokens)
        current_digest = digest(outputs)
        if outputs != baseline:
            mismatch = next(
                index
                for index, (expected, actual) in enumerate(zip(baseline, outputs))
                if expected != actual
            )
            raise RuntimeError(
                f"cycle {cycle} output mismatch at request {mismatch}: "
                f"expected={baseline[mismatch]!r}, actual={outputs[mismatch]!r}"
            )
        request_json(args.base_url, "/health")
        print(
            json.dumps(
                {
                    "event": "cycle",
                    "cycle": cycle,
                    "digest": current_digest,
                    "seconds": round(time.monotonic() - cycle_started, 3),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
