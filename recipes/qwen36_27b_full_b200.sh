#!/usr/bin/env bash
# Qwen3.6-27B full-parameter agent GRPO on 8x B200. Same rollout/agent setup
# as the LoRA recipe; only the optimizer side differs.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export RUN_NAME=${RUN_NAME:-qwen36-27b-full-b200}
export LORA_RANK=0
export LORA_ALPHA=16
export ACTOR_LR=${ACTOR_LR:-1e-6}
# Rollout/train boundary offloads; FSDP runtime CPU offload stays disabled,
# so forward/backward and optimizer steps still run on the GPU.
export ACTOR_PARAM_OFFLOAD=True
export ACTOR_OPTIMIZER_OFFLOAD=True
export REF_PARAM_OFFLOAD=True
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.85}
export CONTEXT_PENALTY_MAX=${CONTEXT_PENALTY_MAX:-0.5}

exec bash "${SCRIPT_DIR}/qwen36_27b_lora_b200.sh" "$@"
