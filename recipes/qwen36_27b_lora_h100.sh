#!/usr/bin/env bash
# Qwen3.6-27B LoRA agent GRPO on 8x H100 (80 GB): TP=8 rollout and Ulysses
# SP=8 training to fit 16K prompt + 48K response.
#
# 32 prompts x 8 rollouts = 256 trajectories per step.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export IMAGE=${IMAGE:-sunnweiwei/verl-qwen35:h100-verl0.8.0-vllm-main-c621af169-mtpaccept-prefixcache-20260614}
export MODEL_PATH=${MODEL_PATH:-/mnt/workspace/models/Qwen3.6-27B}
export RUN_NAME=${RUN_NAME:-qwen36-27b-lora32-h100}

export FSDP_SIZE=8
export SP_SIZE=${SP_SIZE:-8}
export ROLLOUT_TP=${ROLLOUT_TP:-8}
export USE_REMOVE_PADDING=True
export USE_FUSED_KERNELS=True

export LORA_RANK=${LORA_RANK:-32}
export LORA_ALPHA=${LORA_ALPHA:-32}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
export ACTOR_LR=${ACTOR_LR:-5e-5}

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
export ROLLOUT_N=${ROLLOUT_N:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
export MAX_PROMPT_LENGTH=16384
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-49152}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-65536}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1024}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.75}
export AGENT_LOOP_WORKERS=${AGENT_LOOP_WORKERS:-64}
export AGENT_MAX_CONCURRENT_ROLLOUTS=${AGENT_MAX_CONCURRENT_ROLLOUTS:-512}

export ROLLOUT_ENABLE_PREFIX_CACHING=True
export MTP_ENABLE=False
export AGENT_MAX_TURNS=${AGENT_MAX_TURNS:-200}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-120}

exec bash "${SCRIPT_DIR}/../scripts/launch_docker.sh" "$@"
