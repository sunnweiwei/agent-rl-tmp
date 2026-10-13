#!/usr/bin/env bash
# Launch scripts/train.sh in a detached training container.
#
# Host inputs (env):
#   IMAGE            training image (see docker/b200-training, docker/h100-sampling)
#   MODEL_PATH       host model directory (mounted read-only at the same path)
#   DATA_DIR         host directory with TRAIN_FILE / VAL_FILE (default train.parquet / test.parquet)
#   OUTPUT_DIR       host directory for checkpoints, W&B files and trajectories
#   WANDB_ENV_FILE   env file with WANDB_API_KEY (default ../.wandb_env next to the repo)
#   SANDBOX_ENV_FILE optional env file with SANDBOX_MANAGER_URL / SANDBOX_API_TOKEN
#   GPUS             host GPU ids (default 0-7), RUN_NAME, RUN_TAG
# Every other variable read by scripts/train.sh or agents/agents.yaml is
# forwarded when set. Extra arguments are passed to train.sh as Hydra overrides.
#
# Prints the container name on the last line and records it in
# ${OUTPUT_DIR}/container_name.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
REPO_PARENT=$(dirname "${REPO_DIR}")
CONTAINER_REPO=/workspace/agent-rl
DOCKER=${DOCKER:-docker}

: "${IMAGE:?IMAGE is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${DATA_DIR:?DATA_DIR is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
TRAIN_FILE_NAME=${TRAIN_FILE_NAME:-train.parquet}
VAL_FILE_NAME=${VAL_FILE_NAME:-test.parquet}
WANDB_ENV_FILE=${WANDB_ENV_FILE:-${REPO_PARENT}/.wandb_env}
SANDBOX_ENV_FILE=${SANDBOX_ENV_FILE:-${REPO_PARENT}/.sandbox_env}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
RUN_NAME=${RUN_NAME:-agent-grpo}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)-${RUN_NAME}}
CONTAINER_NAME=${CONTAINER_NAME:-agent-rl-${RUN_TAG}}
CACHE_ROOT=${CACHE_ROOT:-${HOME}/.cache}

[ -f "${MODEL_PATH}/config.json" ] || { echo "Missing model config: ${MODEL_PATH}/config.json" >&2; exit 3; }
[ -f "${DATA_DIR}/${TRAIN_FILE_NAME}" ] && [ -f "${DATA_DIR}/${VAL_FILE_NAME}" ] || {
    echo "Missing ${DATA_DIR}/${TRAIN_FILE_NAME} or ${DATA_DIR}/${VAL_FILE_NAME}" >&2; exit 4; }
[ -f "${WANDB_ENV_FILE}" ] || { echo "Missing W&B env file: ${WANDB_ENV_FILE}" >&2; exit 2; }
for config_name in NVIDIA_B200_SHRINK.json NVIDIA_B200_EXPAND_TRUE.json; do
    [ -s "${REPO_DIR}/configs/vllm/moe/${config_name}" ] || {
        echo "Missing vLLM LoRA kernel config: configs/vllm/moe/${config_name}" >&2; exit 1; }
done

ENV_FILE_ARGS=(--env-file "${WANDB_ENV_FILE}")
if [ -f "${SANDBOX_ENV_FILE}" ]; then
    ENV_FILE_ARGS+=(--env-file "${SANDBOX_ENV_FILE}")
    set -a
    # shellcheck disable=SC1090
    source "${SANDBOX_ENV_FILE}"
    set +a
fi
SANDBOX_MANAGER_URL=${SANDBOX_MANAGER_URL:-http://127.0.0.1:8084}
case "${SANDBOX_MANAGER_URL}" in
    http://127.0.0.1:*|http://localhost:*) ;;
    *) [ -n "${SANDBOX_API_TOKEN:-}" ] || {
        echo "SANDBOX_API_TOKEN must be set for non-local SANDBOX_MANAGER_URL=${SANDBOX_MANAGER_URL}" >&2; exit 5; } ;;
esac
export SANDBOX_MANAGER_URL

# CDI renumbers a selected host subset contiguously inside the container.
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
GPU_DEVICE_ARGS=()
CONTAINER_VISIBLE_GPUS=""
for index in "${!GPU_LIST[@]}"; do
    GPU_DEVICE_ARGS+=(--device "nvidia.com/gpu=${GPU_LIST[$index]}")
    CONTAINER_VISIBLE_GPUS+="${CONTAINER_VISIBLE_GPUS:+,}${index}"
done
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-${#GPU_LIST[@]}}

CONTAINER_OUTPUT_DIR=/workspace/outputs/$(basename "${OUTPUT_DIR}")
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${RUN_TAG}}
export TRAJECTORY_TRACE_PATH=${TRAJECTORY_TRACE_PATH:-${CONTAINER_OUTPUT_DIR}/trajectories_${RUN_TAG}.jsonl}

# Forward every train.sh / agents.yaml / preflight variable that is set on the host.
FORWARD_PATTERN='^(TRAINER_|PROJECT_NAME|EXPERIMENT_NAME|CKPTS_DIR|TOTAL_|SAVE_FREQ|TEST_FREQ|VAL_N|VAL_DO_SAMPLE|VAL_TEMPERATURE|VAL_BEFORE_TRAIN|VAL_MAX_SAMPLES|LOG_VAL_|NNODES|NGPUS_PER_NODE|TRAIN_BATCH_SIZE|TRAIN_MAX_SAMPLES|DATA_SEED|MAX_PROMPT_LENGTH|MAX_RESPONSE_LENGTH|DISABLE_THINKING|FSDP_SIZE|SP_SIZE|USE_|FUSED_KERNELS_BACKEND|ATTN_IMPLEMENTATION|LORA_|ACTOR_|REF_|PPO_|LOSS_AGG_MODE|CLIP_RATIO_|KL_LOSS_COEF|ENTROPY_COEFF|ROLLOUT_|MTP_|GROUPED_PARTIAL_|AGENT_|SANDBOX_MANAGER_URL|SANDBOX_PROFILE|PROCESS_REWARDS_ENABLED|PARSE_ERROR_PENALTY|TOOL_ERROR_PENALTY|TOOL_TIMEOUT_PENALTY|ACTION_ERROR_PENALTY_CAP|CONTEXT_PENALTY_|TRAJECTORY_TRACE_PATH|VERL_|VLLM_|FLA_|REQUIRE_VLLM_|WANDB_PROJECT|WANDB_MODE)'
ENV_ARGS=()
while IFS= read -r name; do
    ENV_ARGS+=(-e "${name}")
done < <(compgen -e | grep -E "${FORWARD_PATTERN}" || true)

RESUME_MOUNT_ARGS=()
if [ -n "${TRAINER_RESUME_FROM_PATH:-}" ]; then
    RESUME_MOUNT_ARGS=(-v "${TRAINER_RESUME_FROM_PATH}:${TRAINER_RESUME_FROM_PATH}:ro")
fi

mkdir -p "${OUTPUT_DIR}/wandb" "${OUTPUT_DIR}/ray_tmp"
for cache in huggingface vllm flashinfer torchinductor tilelang triton; do
    mkdir -p "${CACHE_ROOT}/${cache}"
done

"${DOCKER}" run -d \
    --name "${CONTAINER_NAME}" \
    "${GPU_DEVICE_ARGS[@]}" \
    --network host \
    --ipc host \
    --shm-size 256g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --log-opt max-size=2g \
    --log-opt max-file=2 \
    "${ENV_FILE_ARGS[@]}" \
    "${ENV_ARGS[@]}" \
    -v "${REPO_DIR}:${CONTAINER_REPO}" \
    -v "${OUTPUT_DIR}:${CONTAINER_OUTPUT_DIR}" \
    -v "${OUTPUT_DIR}/ray_tmp:/tmp/ray" \
    -v "${DATA_DIR}:/workspace/data:ro" \
    -v "${MODEL_PATH}:${MODEL_PATH}:ro" \
    "${RESUME_MOUNT_ARGS[@]}" \
    -v "${CACHE_ROOT}/huggingface:/root/.cache/huggingface" \
    -v "${CACHE_ROOT}/vllm:/root/.cache/vllm" \
    -v "${CACHE_ROOT}/flashinfer:/root/.cache/flashinfer" \
    -v "${CACHE_ROOT}/torchinductor:/root/.cache/torchinductor" \
    -v "${CACHE_ROOT}/tilelang:/root/.cache/tilelang" \
    -v "${CACHE_ROOT}/triton:/root/.triton" \
    -w "${CONTAINER_REPO}" \
    -e MODEL_PATH="${MODEL_PATH}" \
    -e TRAIN_FILE="/workspace/data/${TRAIN_FILE_NAME}" \
    -e VAL_FILE="/workspace/data/${VAL_FILE_NAME}" \
    -e OUTPUT_DIR="${CONTAINER_OUTPUT_DIR}" \
    -e WANDB_DIR="${CONTAINER_OUTPUT_DIR}/wandb" \
    -e SANDBOX_API_TOKEN \
    -e HYDRA_FULL_ERROR=1 \
    -e TOKENIZERS_PARALLELISM=false \
    -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
    -e RAY_DEDUP_LOGS=0 \
    -e VLLM_LOGGING_LEVEL=INFO \
    -e VLLM_TUNED_CONFIG_FOLDER="${CONTAINER_REPO}/configs/vllm/moe" \
    -e NVIDIA_VISIBLE_DEVICES="${CONTAINER_VISIBLE_GPUS}" \
    -e CUDA_VISIBLE_DEVICES="${CONTAINER_VISIBLE_GPUS}" \
    "${IMAGE}" \
    bash -lc 'set -euo pipefail
        bash scripts/preflight_image.sh
        exec bash scripts/train.sh "$@"' \
    bash "$@" >/dev/null

echo "${CONTAINER_NAME}" > "${OUTPUT_DIR}/container_name"
echo "[launch_docker.sh] logs: ${DOCKER} logs -f ${CONTAINER_NAME}"
echo "[launch_docker.sh] trajectories: ${OUTPUT_DIR}/trajectories_${RUN_TAG}.jsonl"
echo "${CONTAINER_NAME}"
