#!/usr/bin/env bash
# Fail fast inside the training container when a required kernel/vLLM patch is
# missing. Each check is opt-in through an env var set by the recipe.
set -euo pipefail

if [ "${FLA_TILELANG:-}" = "0" ]; then
    python -c "
import inspect, os, fla
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
assert os.environ.get('FLA_TILELANG') == '0'
assert 'cu_seqlens' in inspect.signature(chunk_gated_delta_rule).parameters
print('[preflight] FLA fused GDN backend enabled through Triton (FLA_TILELANG=0, fla=' + fla.__version__ + ')')
"
fi

if [ "${REQUIRE_VLLM_FLA051_FUSED_GDN,,}" = "true" ]; then
    python -c "
import inspect, fla
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import ChunkGatedDeltaRule, fla051_fused_chunk_gated_delta_rule
assert fla.__version__ == '0.5.1'
parameters = inspect.signature(fla051_fused_chunk_gated_delta_rule).parameters
assert 'cu_seqlens' in parameters and 'chunk_indices' in parameters and 'core_attn_out' in parameters
assert 'fla051_fused_chunk_gated_delta_rule' in inspect.getsource(ChunkGatedDeltaRule.forward_native)
print('[preflight] vLLM forward_native uses FLA 0.5.1 fused KKT+solve_tril+recompute_w_u GDN prefill')
"
fi

if [ "${REQUIRE_VLLM_RL_SLEEP_WAKE_FIX,,}" = "true" ]; then
    python -c "
import inspect
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.v1.engine.core import EngineCore
assert 'torch.cuda.synchronize()' in inspect.getsource(CuMemAllocator.sleep)
assert 'torch.cuda.synchronize()' in inspect.getsource(CuMemAllocator.wake_up)
assert 'if not self.model_executor.is_sleeping' in inspect.getsource(EngineCore.wake_up)
print('[preflight] vLLM RL sleep/wake synchronization and partial-wake scheduler gate enabled')
"
fi
