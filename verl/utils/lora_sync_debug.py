import hashlib
import json
import os
from collections.abc import Iterable

import torch


def lora_sync_debug_enabled() -> bool:
    return os.getenv("VERL_LORA_SYNC_DEBUG", "false").lower() in {"1", "true", "yes", "on"}


def fingerprint_lora_tensors(weights: Iterable[tuple[str, torch.Tensor]]) -> dict[str, int | float | str]:
    """Return an exact, order-independent fingerprint for tensors crossing the LoRA sync boundary."""
    digest = hashlib.sha256()
    tensor_count = 0
    total_numel = 0
    total_nbytes = 0
    zero_tensors = 0
    a_zero_tensors = 0
    b_zero_tensors = 0
    max_abs = 0.0

    for name, tensor in sorted(weights, key=lambda item: item[0]):
        if hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()
        tensor = tensor.detach().contiguous()
        raw = tensor.view(torch.uint8).reshape(-1).cpu()
        metadata = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "nbytes": tensor.nbytes,
        }
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(raw.numpy().data)

        tensor_count += 1
        total_numel += tensor.numel()
        total_nbytes += tensor.nbytes
        is_zero = not bool(torch.count_nonzero(raw).item())
        zero_tensors += int(is_zero)
        a_zero_tensors += int(is_zero and ".lora_A." in name)
        b_zero_tensors += int(is_zero and ".lora_B." in name)
        if tensor.numel():
            max_abs = max(max_abs, float(tensor.detach().abs().max().item()))

    return {
        "sha256": digest.hexdigest(),
        "tensor_count": tensor_count,
        "total_numel": total_numel,
        "total_nbytes": total_nbytes,
        "zero_tensors": zero_tensors,
        "a_zero_tensors": a_zero_tensors,
        "b_zero_tensors": b_zero_tensors,
        "max_abs": max_abs,
    }
