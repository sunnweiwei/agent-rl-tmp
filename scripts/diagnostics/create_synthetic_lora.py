#!/usr/bin/env python3
"""Create a shape-exact synthetic PEFT LoRA adapter without loading weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--fill", type=float, default=0.001)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model)
    text_config = getattr(config, "text_config", config)
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        base_model_name_or_path=args.model,
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(text_config)
        model = get_peft_model(model, lora_config)

    meta_state = get_peft_model_state_dict(model)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.peft_config["default"].save_pretrained(output)

    state = {
        name: torch.full(tensor.shape, args.fill, dtype=torch.bfloat16)
        for name, tensor in meta_state.items()
    }
    save_file(state, output / "adapter_model.safetensors")
    print(
        f"saved {len(state)} tensors / {sum(t.numel() for t in state.values())} "
        f"elements to {output}"
    )


if __name__ == "__main__":
    main()
