#!/usr/bin/env python3
"""Compare vLLM's vendored GDN prefill with the FLA 0.5.1 fused path."""

from __future__ import annotations

import argparse
import time

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", default="129,257")
    parser.add_argument("--heads", type=int, default=48)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--skip-reference", action="store_true")
    return parser.parse_args()


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        left.detach().float().flatten(), right.detach().float().flatten(), dim=0
    ).item()


def main() -> None:
    args = _parse_args()
    seq_lens = [int(value) for value in args.seq_lens.split(",")]
    if not seq_lens or any(value <= 0 for value in seq_lens):
        raise ValueError("--seq-lens must contain positive comma-separated integers")

    from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule as vendored_gdn
    from vllm.model_executor.layers.fla.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        fla051_fused_chunk_gated_delta_rule,
    )

    torch.manual_seed(20260718)
    total_tokens = sum(seq_lens)
    shape = (1, total_tokens, args.heads, args.head_dim)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, 64)

    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn_like(q).float(), dim=-1).to(torch.bfloat16)
    v = torch.randn_like(q)
    g = -torch.nn.functional.softplus(
        torch.randn(shape[:-1], device="cuda", dtype=torch.float32)
    ).to(torch.bfloat16)
    beta = torch.sigmoid(
        torch.randn(shape[:-1], device="cuda", dtype=torch.float32)
    ).to(torch.bfloat16)
    initial_state = torch.randn(
        len(seq_lens),
        args.heads,
        args.head_dim,
        args.head_dim,
        device="cuda",
        dtype=torch.float32,
    ) * 0.01

    if not args.skip_reference:
        reference_output, reference_state = vendored_gdn(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state.clone(),
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            use_qk_l2norm_in_kernel=False,
        )
        fused_output, fused_state = fla051_fused_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state.clone(),
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_qk_l2norm_in_kernel=False,
        )
        torch.cuda.synchronize()
        output_cosine = _cosine(reference_output, fused_output)
        state_cosine = _cosine(reference_state, fused_state)
        print(
            f"reference output_cosine={output_cosine:.9f} "
            f"state_cosine={state_cosine:.9f} "
            f"output_max_abs={(reference_output.float() - fused_output.float()).abs().max().item():.6e} "
            f"state_max_abs={(reference_state.float() - fused_state.float()).abs().max().item():.6e}",
            flush=True,
        )
        if output_cosine < 0.9999 or state_cosine < 0.9999:
            raise AssertionError("FLA 0.5.1 fused GDN does not match the vendored path")

    durations = []
    for iteration in range(args.iterations):
        started = time.perf_counter()
        output, final_state = fla051_fused_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_qk_l2norm_in_kernel=False,
        )
        torch.cuda.synchronize()
        duration = time.perf_counter() - started
        durations.append(duration)
        if not torch.isfinite(output).all() or not torch.isfinite(final_state).all():
            raise AssertionError(f"non-finite fused GDN output at iteration {iteration}")
        print(f"iteration={iteration} seconds={duration:.6f}", flush=True)

    steady_state = durations[1:] or durations
    print(
        f"device={torch.cuda.get_device_name()} total_tokens={total_tokens} "
        f"sequences={len(seq_lens)} steady_state_mean={sum(steady_state) / len(steady_state):.6f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
