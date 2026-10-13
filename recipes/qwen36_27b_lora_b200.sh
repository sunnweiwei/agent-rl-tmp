#!/usr/bin/env bash
# Qwen3.6-27B LoRA agent GRPO on 8x B200 (the validated packed/fused path).
#
#   MODEL_PATH=... DATA_DIR=... OUTPUT_DIR=... bash recipes/qwen36_27b_lora_b200.sh
#
# 16 prompts x 16 rollouts, 4 PPO mini-batches x 1 epoch = 4 optimizer updates
# per rollout batch. 16K prompt + 48K response, thinking disabled.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# lora64fix adds the vLLM LoRA int64 row-index patch: with 27B's 34,816-wide
# gate/up projection, int32 offsets overflow once a packed batch exceeds
# 61,680 tokens (this recipe batches up to 131,072).
export IMAGE=${IMAGE:-sunnweiwei/verl-qwen35:b200-verl0.8.0-vllm-c621af169-fla051-fused-intra-rl-sleepfix-lora64fix-20260719}
export MODEL_PATH=${MODEL_PATH:-/mnt/workspace/models/Qwen3.6-27B}
export RUN_NAME=${RUN_NAME:-qwen36-27b-lora32-b200}

# Do not fall back from these: FLA fused GDN, fused kernels, remove-padding,
# and the patched vLLM sleep/wake path. SP stays 1 (Ulysses SP=8 hit a
# misaligned-address failure on B200 with FLA context parallelism).
export REQUIRE_VLLM_FLA051_FUSED_GDN=True
export REQUIRE_VLLM_RL_SLEEP_WAKE_FIX=True
export FLA_TILELANG=0
export ROLLOUT_ATTENTION_BACKEND=${ROLLOUT_ATTENTION_BACKEND:-TRITON_ATTN}
export USE_REMOVE_PADDING=True
export USE_FUSED_KERNELS=True
export SP_SIZE=1
export FSDP_SIZE=8
export ROLLOUT_TP=1

export LORA_RANK=${LORA_RANK:-32}
export LORA_ALPHA=${LORA_ALPHA:-64}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
export ACTOR_LR=${ACTOR_LR:-5e-5}

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export ROLLOUT_N=${ROLLOUT_N:-16}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
export PPO_EPOCHS=${PPO_EPOCHS:-1}
export MAX_PROMPT_LENGTH=16384
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-49152}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-65536}
# Dynamic batching must fit the longest single sequence (16K + 48K).
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-65536}

# At most 256 train / VAL_N x val-size trajectories in flight; 512 avoids
# scheduler throttling. B200 memory goes to KV cache and prefill batch.
export AGENT_LOOP_WORKERS=${AGENT_LOOP_WORKERS:-32}
export AGENT_MAX_CONCURRENT_ROLLOUTS=${AGENT_MAX_CONCURRENT_ROLLOUTS:-512}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-512}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-131072}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.90}

# Prefix caching is required for long shared agent prompts. MTP was correct
# on B200 but reduced cache reuse and throughput, so it stays off.
export ROLLOUT_ENABLE_PREFIX_CACHING=True
export MTP_ENABLE=False

export PROCESS_REWARDS_ENABLED=${PROCESS_REWARDS_ENABLED:-True}
export PARSE_ERROR_PENALTY=${PARSE_ERROR_PENALTY:-0.25}
export TOOL_ERROR_PENALTY=${TOOL_ERROR_PENALTY:-0.05}
export TOOL_TIMEOUT_PENALTY=${TOOL_TIMEOUT_PENALTY:-0.05}
export ACTION_ERROR_PENALTY_CAP=${ACTION_ERROR_PENALTY_CAP:-0.5}
export CONTEXT_PENALTY_BUFFER_TOKENS=${CONTEXT_PENALTY_BUFFER_TOKENS:-8192}
export CONTEXT_PENALTY_MAX=${CONTEXT_PENALTY_MAX:-1.0}
export AGENT_MAX_TURNS=${AGENT_MAX_TURNS:-200}

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
export TEST_FREQ=${TEST_FREQ:-5}
export SAVE_FREQ=${SAVE_FREQ:--1}  # set e.g. 10 for resumable runs

exec bash "${SCRIPT_DIR}/../scripts/launch_docker.sh" "$@"
