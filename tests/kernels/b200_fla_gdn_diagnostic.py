#!/usr/bin/env python3
"""Exercise FLA packed GDN forward/backward with the Qwen3.6-27B shape."""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.checkpoint import checkpoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", default="257,511,1024")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=64, choices=(16, 32, 64))
    parser.add_argument("--query-heads", type=int, default=48)
    parser.add_argument("--value-heads", type=int, default=48)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--reference-check", action="store_true")
    return parser.parse_args()


def _call_fla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        chunk_size=chunk_size,
    )[0]


def _run_once(
    seq_lens: list[int], chunk_size: int, query_heads: int, value_heads: int, use_checkpoint: bool
) -> tuple[float, float]:
    total_tokens = sum(seq_lens)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Qwen3.6 has 16 Q/K heads and 48 V heads, then repeat_interleaves Q/K
    # to 48 immediately before calling FLA. query_heads=48 tests that exact call.
    q = torch.randn(1, total_tokens, query_heads, 128, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, total_tokens, query_heads, 128, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(1, total_tokens, value_heads, 128, device=device, dtype=dtype, requires_grad=True)
    g_logits = torch.randn(1, total_tokens, value_heads, device=device, dtype=torch.float32, requires_grad=True)
    beta_logits = torch.randn(1, total_tokens, value_heads, device=device, dtype=torch.float32, requires_grad=True)
    g = -torch.nn.functional.softplus(g_logits).to(dtype)
    beta = beta_logits.sigmoid().to(dtype)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()], device=device, dtype=torch.long
    )

    torch.cuda.synchronize()
    print("stage=forward_start", flush=True)
    started = time.perf_counter()
    if use_checkpoint:
        output = checkpoint(
            lambda q_, k_, v_, g_, beta_: _call_fla(
                q_, k_, v_, g_, beta_, cu_seqlens, chunk_size
            ),
            q,
            k,
            v,
            g,
            beta,
            use_reentrant=False,
        )
    else:
        output = _call_fla(q, k, v, g, beta, cu_seqlens, chunk_size)
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - started
    print(f"stage=forward_done seconds={forward_seconds:.6f}", flush=True)

    loss = output.float().square().mean()
    print("stage=backward_start", flush=True)
    started = time.perf_counter()
    grads = torch.autograd.grad(loss, (q, k, v, g_logits, beta_logits))
    torch.cuda.synchronize()
    backward_seconds = time.perf_counter() - started
    print(f"stage=backward_done seconds={backward_seconds:.6f}", flush=True)

    if output.shape != v.shape:
        raise AssertionError(f"unexpected output shape: {output.shape} != {v.shape}")
    if not torch.isfinite(output).all():
        raise AssertionError("non-finite GDN output")
    for name, grad in zip(("q", "k", "v", "g", "beta"), grads, strict=True):
        if not torch.isfinite(grad).all():
            raise AssertionError(f"non-finite {name} gradient")
    return forward_seconds, backward_seconds


def _reference_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    seq_lens: list[int],
    chunk_size: int,
) -> torch.Tensor:
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    outputs = []
    offset = 0
    for seq_len in seq_lens:
        seq_slice = slice(offset, offset + seq_len)
        outputs.append(
            torch_chunk_gated_delta_rule(
                q[:, seq_slice],
                k[:, seq_slice],
                v[:, seq_slice],
                g[:, seq_slice],
                beta[:, seq_slice],
                chunk_size=chunk_size,
                use_qk_l2norm_in_kernel=True,
            )[0]
        )
        offset += seq_len
    return torch.cat(outputs, dim=1)


def _cosine_similarity(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_float = actual.detach().float().flatten()
    expected_float = expected.detach().float().flatten()
    return torch.nn.functional.cosine_similarity(actual_float, expected_float, dim=0).item()


def _run_reference_check(seq_lens: list[int], chunk_size: int, query_heads: int, value_heads: int) -> None:
    if query_heads != value_heads:
        raise ValueError("--reference-check requires matching query and value head counts")

    torch.manual_seed(20260716)
    total_tokens = sum(seq_lens)
    device = torch.device("cuda")
    q = torch.randn(1, total_tokens, query_heads, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, total_tokens, query_heads, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, total_tokens, value_heads, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    g_logits = torch.randn(1, total_tokens, value_heads, device=device, dtype=torch.float32, requires_grad=True)
    beta_logits = torch.randn(1, total_tokens, value_heads, device=device, dtype=torch.float32, requires_grad=True)
    g = -torch.nn.functional.softplus(g_logits).to(torch.bfloat16)
    beta = beta_logits.sigmoid().to(torch.bfloat16)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()], device=device, dtype=torch.long
    )

    fla_output = _call_fla(q, k, v, g, beta, cu_seqlens, chunk_size)
    reference_output = _reference_packed(q, k, v, g, beta, seq_lens, chunk_size)
    loss_weight = torch.randn_like(fla_output)
    inputs = (q, k, v, g_logits, beta_logits)
    fla_grads = torch.autograd.grad((fla_output.float() * loss_weight.float()).mean(), inputs, retain_graph=True)
    reference_grads = torch.autograd.grad(
        (reference_output.float() * loss_weight.float()).mean(), inputs
    )

    output_diff = (fla_output.float() - reference_output.float()).abs()
    output_cosine = _cosine_similarity(fla_output, reference_output)
    print(
        "reference output "
        f"max_abs={output_diff.max().item():.6e} "
        f"mean_abs={output_diff.mean().item():.6e} cosine={output_cosine:.8f}"
    )
    if output_cosine < 0.999:
        raise AssertionError(f"FLA/reference output cosine is too low: {output_cosine}")

    for name, fla_grad, reference_grad in zip(
        ("q", "k", "v", "g", "beta"), fla_grads, reference_grads, strict=True
    ):
        grad_diff = (fla_grad.float() - reference_grad.float()).abs()
        grad_cosine = _cosine_similarity(fla_grad, reference_grad)
        print(
            f"reference grad={name} max_abs={grad_diff.max().item():.6e} "
            f"mean_abs={grad_diff.mean().item():.6e} cosine={grad_cosine:.8f}"
        )
        if grad_cosine < 0.99:
            raise AssertionError(f"FLA/reference {name} gradient cosine is too low: {grad_cosine}")


def main() -> None:
    args = _parse_args()
    seq_lens = [int(value) for value in args.seq_lens.split(",")]
    if not seq_lens or any(value <= 0 for value in seq_lens):
        raise ValueError("--seq-lens must contain positive comma-separated integers")

    import fla
    import triton

    print(
        f"device={torch.cuda.get_device_name()} torch={torch.__version__} "
        f"cuda={torch.version.cuda} triton={triton.__version__} "
        f"fla={getattr(fla, '__version__', 'unknown')} seq_lens={seq_lens} "
        f"query_heads={args.query_heads} value_heads={args.value_heads} checkpoint={args.checkpoint} "
        f"FLA_TILELANG={__import__('os').environ.get('FLA_TILELANG', '<unset>')}"
    )
    if args.reference_check:
        _run_reference_check(seq_lens, args.chunk_size, args.query_heads, args.value_heads)
    for iteration in range(args.iterations):
        forward_seconds, backward_seconds = _run_once(
            seq_lens, args.chunk_size, args.query_heads, args.value_heads, args.checkpoint
        )
        print(
            f"iteration={iteration} forward={forward_seconds:.6f}s "
            f"backward={backward_seconds:.6f}s"
        )


if __name__ == "__main__":
    main()
