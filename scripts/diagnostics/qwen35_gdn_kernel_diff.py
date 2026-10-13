#!/usr/bin/env python3
"""Compare Qwen3.5 chunk-prefill and recurrent-decode GDN kernels."""

import argparse

import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[16, 128, 1024, 8192])
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.inference_mode()
def compare(length: int, heads: int, key_dim: int, value_dim: int, seed: int):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    q = torch.randn(1, length, heads, key_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(1, length, heads, value_dim, device=device, dtype=dtype)
    a = torch.randn(1, length, heads, device=device, dtype=dtype)
    b = torch.randn_like(a)
    a_log = torch.randn(heads, device=device, dtype=torch.float32) * 0.2
    dt_bias = torch.randn(heads, device=device, dtype=torch.float32) * 0.2

    g = -a_log.exp() * F.softplus(a.float() + dt_bias)
    beta = b.sigmoid()
    initial_state = torch.zeros(2, heads, value_dim, key_dim, device=device, dtype=torch.float32)

    chunk_out, _ = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state[1:].clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    recurrent_state = initial_state.clone()
    recurrent_parts = []
    cu_seqlens = torch.tensor([0, 1], device=device, dtype=torch.int32)
    state_indices = torch.ones(1, device=device, dtype=torch.int32)
    for pos in range(length):
        token_out, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=a_log,
            a=a[:, pos : pos + 1],
            b=b[:, pos : pos + 1],
            dt_bias=dt_bias,
            q=q[:, pos : pos + 1],
            k=k[:, pos : pos + 1],
            v=v[:, pos : pos + 1],
            initial_state=recurrent_state,
            inplace_final_state=True,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )
        recurrent_parts.append(token_out)
    recurrent_out = torch.cat(recurrent_parts, dim=1)

    if recurrent_out.shape != chunk_out.shape:
        raise RuntimeError(f"kernel output shape mismatch: chunk={chunk_out.shape}, recurrent={recurrent_out.shape}")

    lhs = chunk_out.float()
    rhs = recurrent_out.float()
    diff = rhs - lhs
    denom = lhs.square().mean().sqrt().clamp_min(1e-12)
    checkpoints = sorted({0, length // 4, length // 2, 3 * length // 4, length - 1})
    checkpoint_mae = {
        str(pos + 1): (rhs[:, pos] - lhs[:, pos]).abs().mean().item() for pos in checkpoints
    }
    return {
        "length": length,
        "chunk_rms": lhs.square().mean().sqrt().item(),
        "mae": diff.abs().mean().item(),
        "rmse": diff.square().mean().sqrt().item(),
        "relative_rmse": (diff.square().mean().sqrt() / denom).item(),
        "max_abs": diff.abs().max().item(),
        "cosine": F.cosine_similarity(lhs.flatten(), rhs.flatten(), dim=0).item(),
        "checkpoint_mae": checkpoint_mae,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for length in args.lengths:
        result = compare(length, args.heads, args.key_dim, args.value_dim, args.seed)
        fields = [
            f"length={result['length']}",
            f"mae={result['mae']:.8g}",
            f"rmse={result['rmse']:.8g}",
            f"relative_rmse={result['relative_rmse']:.8g}",
            f"max_abs={result['max_abs']:.8g}",
            f"cosine={result['cosine']:.10f}",
            f"checkpoint_mae={result['checkpoint_mae']}",
        ]
        print(" ".join(fields), flush=True)


if __name__ == "__main__":
    main()
