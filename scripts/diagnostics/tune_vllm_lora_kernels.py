#!/usr/bin/env python3
"""Tune vLLM dense LoRA Triton kernels for the Qwen3.5-9B B200 recipe."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

from vllm.lora.ops.triton_ops import LoRAKernelMeta, lora_expand, lora_shrink
from vllm.lora.ops.triton_ops import lora_expand_op, lora_shrink_op
from vllm.lora.ops.triton_ops.utils import (
    _LORA_A_PTR_DICT,
    _LORA_B_PTR_DICT,
)


Config = dict[str, int | None]

M_VALUES = (1, 64, 256, 512, 2048, 8192, 32768, 131072)
SHRINK_K_VALUES = (4096, 12288)
EXPAND_N_VALUES = (32, 1024, 4096, 8192, 12288)
NUM_SLICES = (1, 2, 3, 4)
MAX_LORAS_WITH_SENTINEL = 2
RANK = 32


def shrink_candidates(m: int) -> list[Config]:
    default_split_k = 64 if m < 128 else 8
    default_block_k = 256 if m < 128 else 32
    candidates: list[Config] = [
        {
            "block_m": 32,
            "block_n": 16,
            "block_k": default_block_k,
            "split_k": default_split_k,
            "num_warps": 4,
            "num_ctas": 1,
            "group_size_m": 8,
            "num_stages": 2,
            "max_nreg": None,
        }
    ]
    layouts = (
        (16, 32, 64, 32, 4, 2),
        (32, 32, 64, 16, 4, 2),
        (32, 32, 128, 16, 4, 2),
        (32, 32, 256, 8, 4, 2),
        (64, 32, 64, 8, 4, 2),
        (64, 32, 128, 4, 4, 2),
        (64, 32, 256, 2, 8, 2),
        (128, 32, 64, 4, 4, 3),
        (128, 32, 128, 2, 8, 3),
        (128, 32, 256, 1, 8, 3),
    )
    for block_m, block_n, block_k, split_k, num_warps, num_stages in layouts:
        candidates.append(
            {
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "split_k": split_k,
                "num_warps": num_warps,
                "num_ctas": 1,
                "group_size_m": 8,
                "num_stages": num_stages,
                "max_nreg": None,
            }
        )
    return candidates


def expand_candidates() -> list[Config]:
    layouts = (
        (16, 32, 16, 4, 2),
        (16, 64, 32, 4, 2),
        (32, 64, 16, 4, 2),
        (32, 64, 32, 4, 3),
        (32, 128, 32, 4, 3),
        (64, 64, 16, 4, 2),
        (64, 64, 32, 4, 3),
        (64, 128, 32, 4, 3),
        (64, 256, 32, 8, 3),
        (128, 64, 32, 4, 3),
        (128, 128, 32, 8, 3),
        (128, 256, 32, 8, 4),
    )
    return [
        {
            "block_m": block_m,
            "block_n": block_n,
            "block_k": block_k,
            "num_warps": num_warps,
            "num_ctas": 1,
            "num_stages": num_stages,
            "max_nreg": None,
        }
        for block_m, block_n, block_k, num_warps, num_stages in layouts
    ]


def make_meta(m: int) -> LoRAKernelMeta:
    meta = LoRAKernelMeta.make(max_loras=1, max_num_tokens=m, device="cuda")
    meta.prepare_tensors(torch.zeros(m, dtype=torch.int32, device="cuda"))
    return meta


def benchmark(call: Any, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return statistics.median(samples)


def set_nested(
    root: dict[str, Any],
    max_loras: int,
    num_slices: int,
    m: int,
    k: int,
    n: int,
    config: Config,
) -> None:
    root.setdefault(str(max_loras), {}).setdefault(str(num_slices), {}).setdefault(
        str(m), {}
    ).setdefault(str(k), {})[str(n)] = config


def tune_shrink(m: int, k: int) -> tuple[Config, float]:
    torch.manual_seed(0)
    inputs = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weights = [
        torch.randn((1, RANK, k), dtype=torch.bfloat16, device="cuda")
    ]
    output = torch.empty((1, m, RANK), dtype=torch.float32, device="cuda")
    meta = make_meta(m)
    meta_args = meta.meta_args(m, specialize_active_lora=True)
    results: list[tuple[float, Config]] = []

    for config in shrink_candidates(m):
        lora_shrink_op.get_lora_op_configs = lambda *args, _c=config, **kwargs: _c
        _LORA_A_PTR_DICT.clear()

        def call() -> None:
            lora_shrink(inputs, weights, output, *meta_args, 1.0)

        try:
            elapsed = benchmark(call, warmup=2, iterations=10 if m < 32768 else 4)
        except Exception as exc:
            print(f"shrink rejected m={m} k={k} config={config}: {exc}", flush=True)
            continue
        if math.isfinite(elapsed):
            results.append((elapsed, config))

    if not results:
        raise RuntimeError(f"No valid shrink config for m={m}, k={k}")
    elapsed, config = min(results, key=lambda item: item[0])
    print(
        f"shrink m={m:6d} k={k:5d}: {elapsed:.4f} ms {config}",
        flush=True,
    )
    return config, elapsed


def tune_expand(m: int, n: int) -> tuple[Config, float]:
    torch.manual_seed(0)
    inputs = torch.randn((1, m, RANK), dtype=torch.float32, device="cuda")
    weights = [
        torch.randn((1, n, RANK), dtype=torch.bfloat16, device="cuda")
    ]
    output = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")
    meta = make_meta(m)
    meta_args = meta.meta_args(m, specialize_active_lora=True)
    results: list[tuple[float, Config]] = []

    for config in expand_candidates():
        lora_expand_op.get_lora_op_configs = lambda *args, _c=config, **kwargs: _c
        _LORA_B_PTR_DICT.clear()

        def call() -> None:
            lora_expand(inputs, weights, output, *meta_args, 0, True)

        try:
            elapsed = benchmark(call, warmup=2, iterations=10 if m < 32768 else 4)
        except Exception as exc:
            print(f"expand rejected m={m} n={n} config={config}: {exc}", flush=True)
            continue
        if math.isfinite(elapsed):
            results.append((elapsed, config))

    if not results:
        raise RuntimeError(f"No valid expand config for m={m}, n={n}")
    elapsed, config = min(results, key=lambda item: item[0])
    print(
        f"expand m={m:6d} n={n:5d}: {elapsed:.4f} ms {config}",
        flush=True,
    )
    return config, elapsed


def validate_configs(shrink: Config, expand: Config) -> None:
    m, k, n = 37, 4096, 1024
    meta = make_meta(m)
    meta_args = meta.meta_args(m, specialize_active_lora=True)

    for num_slices in (1, 4):
        x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        a = [
            torch.randn((1, RANK, k), dtype=torch.bfloat16, device="cuda")
            for _ in range(num_slices)
        ]
        actual_a = torch.empty(
            (num_slices, m, RANK), dtype=torch.float32, device="cuda"
        )
        lora_shrink_op.get_lora_op_configs = lambda *args, **kwargs: shrink
        _LORA_A_PTR_DICT.clear()
        lora_shrink(x, a, actual_a, *meta_args, 1.0)
        expected_a = torch.stack(
            [torch.nn.functional.linear(x, weight[0]).float() for weight in a]
        )
        torch.testing.assert_close(actual_a, expected_a, rtol=0.06, atol=0.06)

        b = [
            torch.randn((1, n, RANK), dtype=torch.bfloat16, device="cuda")
            for _ in range(num_slices)
        ]
        base = torch.randn(
            (m, n * num_slices), dtype=torch.bfloat16, device="cuda"
        )
        actual_b = base.clone()
        lora_expand_op.get_lora_op_configs = lambda *args, **kwargs: expand
        _LORA_B_PTR_DICT.clear()
        lora_expand(actual_a, b, actual_b, *meta_args, 0, True)
        expected_b = base.clone()
        for slice_idx in range(num_slices):
            start = slice_idx * n
            expected_b[:, start : start + n] += torch.nn.functional.linear(
                expected_a[slice_idx].bfloat16(), b[slice_idx][0]
            )
        torch.testing.assert_close(actual_b, expected_b, rtol=0.06, atol=0.06)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if torch.cuda.get_device_name() != "NVIDIA B200":
        raise RuntimeError(f"Expected NVIDIA B200, got {torch.cuda.get_device_name()}")

    shrink_data: dict[str, Any] = {}
    expand_data: dict[str, Any] = {}
    selected_shrink: Config | None = None
    selected_expand: Config | None = None

    for m in M_VALUES:
        for k in SHRINK_K_VALUES:
            config, _ = tune_shrink(m, k)
            selected_shrink = config
            for num_slices in NUM_SLICES:
                set_nested(
                    shrink_data,
                    MAX_LORAS_WITH_SENTINEL,
                    num_slices,
                    m,
                    k,
                    RANK,
                    config,
                )

    for m in M_VALUES:
        for n in EXPAND_N_VALUES:
            config, _ = tune_expand(m, n)
            selected_expand = config
            for num_slices in NUM_SLICES:
                set_nested(
                    expand_data,
                    MAX_LORAS_WITH_SENTINEL,
                    num_slices,
                    m,
                    RANK,
                    n,
                    config,
                )

    assert selected_shrink is not None and selected_expand is not None
    validate_configs(selected_shrink, selected_expand)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "NVIDIA_B200_SHRINK.json": shrink_data,
        "NVIDIA_B200_EXPAND_TRUE.json": expand_data,
    }
    for name, data in outputs.items():
        path = args.output_dir / name
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(f"wrote {path}", flush=True)
    print("correctness validation passed", flush=True)


if __name__ == "__main__":
    main()
