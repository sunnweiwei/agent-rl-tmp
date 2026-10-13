#!/usr/bin/env python3
"""Run a packed Qwen3.6 GDN layer and LoRA backward on B200."""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/mnt/workspace/models/Qwen3.6-27B")
    parser.add_argument("--seq-lens", default="73,129")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    if os.environ.get("FLA_TILELANG") != "0":
        raise RuntimeError("This B200 diagnostic requires FLA_TILELANG=0")

    args = _parse_args()
    seq_lens = [int(value) for value in args.seq_lens.split(",")]
    if not seq_lens or any(value <= 0 for value in seq_lens):
        raise ValueError("--seq-lens must contain positive comma-separated integers")

    from peft import LoraConfig, inject_adapter_in_model
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    from verl.models.transformers.qwen3_5 import qwen3_5_gated_delta_net_forward

    config = AutoConfig.from_pretrained(args.model_path).text_config
    layer = Qwen3_5GatedDeltaNet(config, layer_idx=0)
    layer.forward = qwen3_5_gated_delta_net_forward.__get__(layer, type(layer))
    layer.requires_grad_(False)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules={"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"},
        bias="none",
    )
    layer = inject_adapter_in_model(lora_config, layer)
    layer = layer.to(device="cuda", dtype=torch.bfloat16)

    gdn_backend = layer.chunk_gated_delta_rule
    backend_params = inspect.signature(gdn_backend).parameters
    if "cu_seqlens" not in backend_params:
        raise AssertionError(f"GDN backend does not expose packed sequences: {gdn_backend}")
    if not gdn_backend.__module__.startswith("fla."):
        raise AssertionError(f"expected FLA GDN backend, got {gdn_backend.__module__}")

    total_tokens = sum(seq_lens)
    hidden_states = torch.randn(
        1,
        total_tokens,
        config.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    cu_seqlens_cpu = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.long)
    cu_seqlens = cu_seqlens_cpu.to("cuda")

    output = checkpoint(
        lambda states: layer(
            states,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        ),
        hidden_states,
        use_reentrant=False,
    )
    output.float().square().mean().backward()
    torch.cuda.synchronize()

    if output.shape != hidden_states.shape:
        raise AssertionError(f"unexpected output shape: {output.shape} != {hidden_states.shape}")
    if not torch.isfinite(output).all() or not torch.isfinite(hidden_states.grad).all():
        raise AssertionError("non-finite layer output or input gradient")

    trainable = [(name, parameter) for name, parameter in layer.named_parameters() if parameter.requires_grad]
    missing = [name for name, parameter in trainable if parameter.grad is None]
    non_finite = [
        name for name, parameter in trainable if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    nonzero = [
        name for name, parameter in trainable if parameter.grad is not None and torch.count_nonzero(parameter.grad)
    ]
    if missing or non_finite or not nonzero:
        raise AssertionError(
            f"invalid LoRA gradients: missing={missing} non_finite={non_finite} nonzero={len(nonzero)}"
        )

    print(
        f"device={torch.cuda.get_device_name()} backend={gdn_backend.__module__} "
        f"seq_lens={seq_lens} hidden_size={config.hidden_size} "
        f"k_heads={config.linear_num_key_heads} v_heads={config.linear_num_value_heads} "
        f"trainable_lora_tensors={len(trainable)} nonzero_grad_tensors={len(nonzero)} "
        f"output_abs_mean={output.float().abs().mean().item():.6e}"
    )


if __name__ == "__main__":
    main()
