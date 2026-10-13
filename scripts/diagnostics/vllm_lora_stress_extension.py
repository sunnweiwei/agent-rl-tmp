"""vLLM worker extension for reproducing VERL tensor-LoRA hot reloads."""

from __future__ import annotations

import json
import time

import torch
from safetensors.torch import load_file

from verl.utils.vllm import TensorLoRARequest, VLLMHijack
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
)

# Apply the same LoRA manager patch used by the production colocated worker
# before vLLM constructs its model manager.
VLLMHijack.hijack()


class SyntheticLoRAStressWorkerExtension:
    """Expose the production tensor-LoRA loader through dev collective RPC."""

    def reload_tensor_lora(self, serialized_request: str) -> dict[str, object]:
        request = json.loads(serialized_request)
        started = time.monotonic()
        adapter_path = request["adapter_path"]
        lora_int_id = int(request.get("lora_int_id", VLLM_LORA_INT_ID))
        lora_name = request.get("lora_name", VLLM_LORA_NAME)

        loaded_loras = self.list_loras()
        if lora_int_id in loaded_loras:
            self.remove_lora(lora_int_id)

        with open(f"{adapter_path}/adapter_config.json", encoding="utf-8") as file:
            peft_config = json.load(file)
        tensors = load_file(f"{adapter_path}/adapter_model.safetensors", device="cpu")
        lora_request = TensorLoRARequest(
            lora_name=lora_name,
            lora_int_id=lora_int_id,
            lora_path=VLLM_LORA_PATH,
            peft_config=peft_config,
            lora_tensors=tensors,
        )
        added = self.add_lora(lora_request)
        torch.cuda.synchronize()
        return {
            "added": bool(added),
            "loaded_loras": sorted(self.list_loras()),
            "tensor_count": len(tensors),
            "total_numel": sum(tensor.numel() for tensor in tensors.values()),
            "seconds": round(time.monotonic() - started, 3),
        }
