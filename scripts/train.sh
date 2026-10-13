#!/usr/bin/env bash
# Agent GRPO trainer entrypoint. Runs inside the training container (see
# scripts/launch_docker.sh); every setting below is an env var with a default,
# and extra arguments are passed through as Hydra overrides.
#
# Required: MODEL_PATH, TRAIN_FILE, VAL_FILE, OUTPUT_DIR, WANDB_API_KEY.

set -euo pipefail

is_true() {
    [[ "${1:-}" =~ ^([Tt]rue|TRUE|1)$ ]]
}

for name in MODEL_PATH TRAIN_FILE VAL_FILE OUTPUT_DIR; do
    if [ -z "${!name:-}" ]; then
        echo "${name} is required." >&2
        exit 2
    fi
done
[ -f "${MODEL_PATH}/config.json" ] || { echo "Missing model config: ${MODEL_PATH}/config.json" >&2; exit 3; }
[ -f "${TRAIN_FILE}" ] && [ -f "${VAL_FILE}" ] || { echo "Missing ${TRAIN_FILE} or ${VAL_FILE}" >&2; exit 4; }
[ -n "${WANDB_API_KEY:-}" ] || { echo "WANDB_API_KEY is required." >&2; exit 2; }

# ---- trainer / logging ------------------------------------------------------
TRAINER_MODULE=${TRAINER_MODULE:-verl.trainer.main_ppo_sync}
PROJECT_NAME=${PROJECT_NAME:-agent-rl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-agent-grpo-$(date +%Y%m%d_%H%M%S)}
CKPTS_DIR=${CKPTS_DIR:-${OUTPUT_DIR%/}/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:-5}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}
TRAINER_RESUME_MODE=${TRAINER_RESUME_MODE:-disable}
TRAINER_RESUME_FROM_PATH=${TRAINER_RESUME_FROM_PATH:-}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# ---- data -------------------------------------------------------------------
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
DATA_SEED=${DATA_SEED:-42}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-16384}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-49152}
DISABLE_THINKING=${DISABLE_THINKING:-True}

# ---- model / FSDP -----------------------------------------------------------
FSDP_SIZE=${FSDP_SIZE:-8}
SP_SIZE=${SP_SIZE:-1}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-True}
FUSED_KERNELS_BACKEND=${FUSED_KERNELS_BACKEND:-torch}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
LORA_MERGE=${LORA_MERGE:-False}
# Stage-boundary offloads: trainer state leaves the GPU while vLLM rolls out.
# offload_policy (FSDP runtime CPU offload during compute) must stay False.
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}

# ---- optimization -----------------------------------------------------------
ACTOR_LR=${ACTOR_LR:-5e-5}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
PPO_EPOCHS=${PPO_EPOCHS:-1}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-65536}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-seq-mean-token-mean}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
USE_KL_LOSS=${USE_KL_LOSS:-False}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.01}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
ACTOR_CALCULATE_ENTROPY=${ACTOR_CALCULATE_ENTROPY:-True}

# ---- rollout (vLLM) ---------------------------------------------------------
ROLLOUT_N=${ROLLOUT_N:-16}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.90}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-131072}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-512}
ROLLOUT_ENABLE_PREFIX_CACHING=${ROLLOUT_ENABLE_PREFIX_CACHING:-True}
ROLLOUT_DISABLE_LOG_STATS=${ROLLOUT_DISABLE_LOG_STATS:-False}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
ROLLOUT_GDN_PREFILL_BACKEND=${ROLLOUT_GDN_PREFILL_BACKEND:-triton}
ROLLOUT_ATTENTION_BACKEND=${ROLLOUT_ATTENTION_BACKEND:-}
if (( LORA_RANK > 0 )); then
    ROLLOUT_LOAD_FORMAT=${ROLLOUT_LOAD_FORMAT:-safetensors}
    ROLLOUT_LAYERED_SUMMON=${ROLLOUT_LAYERED_SUMMON:-True}
else
    ROLLOUT_LOAD_FORMAT=${ROLLOUT_LOAD_FORMAT:-dummy}
    ROLLOUT_LAYERED_SUMMON=${ROLLOUT_LAYERED_SUMMON:-False}
fi
VAL_N=${VAL_N:-4}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-False}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}

MTP_ENABLE=${MTP_ENABLE:-False}
MTP_ENABLE_ROLLOUT=${MTP_ENABLE_ROLLOUT:-${MTP_ENABLE}}
MTP_ENABLE_TRAIN=${MTP_ENABLE_TRAIN:-False}
MTP_NUM_SPECULATIVE_TOKENS=${MTP_NUM_SPECULATIVE_TOKENS:-0}
MTP_DRAFT_ATTENTION_BACKEND=${MTP_DRAFT_ATTENTION_BACKEND:-${ROLLOUT_ATTENTION_BACKEND}}

# Behavior log-probs come from vLLM; bypass mode trains on them directly.
ROLLOUT_CALCULATE_LOG_PROBS=${ROLLOUT_CALCULATE_LOG_PROBS:-True}
ACTOR_USE_ROLLOUT_LOG_PROBS=${ACTOR_USE_ROLLOUT_LOG_PROBS:-${ROLLOUT_CALCULATE_LOG_PROBS}}
ROLLOUT_CORRECTION_BYPASS_MODE=${ROLLOUT_CORRECTION_BYPASS_MODE:-True}
ROLLOUT_CORRECTION_LOSS_TYPE=${ROLLOUT_CORRECTION_LOSS_TYPE:-ppo_clip}
ROLLOUT_CORRECTION_IS=${ROLLOUT_CORRECTION_IS:-null}
ROLLOUT_CORRECTION_IS_THRESHOLD=${ROLLOUT_CORRECTION_IS_THRESHOLD:-2.0}
ROLLOUT_CORRECTION_IS_BATCH_NORMALIZE=${ROLLOUT_CORRECTION_IS_BATCH_NORMALIZE:-False}
ROLLOUT_CORRECTION_PRE_UPDATE_METRIC_MODE=${ROLLOUT_CORRECTION_PRE_UPDATE_METRIC_MODE:-first_minibatch}
if is_true "${ROLLOUT_CORRECTION_BYPASS_MODE}"; then
    ACTOR_POLICY_LOSS_MODE=${ACTOR_POLICY_LOSS_MODE:-bypass_mode}
else
    ACTOR_POLICY_LOSS_MODE=${ACTOR_POLICY_LOSS_MODE:-vanilla}
fi

# Grouped partial rollout: start training once enough prompt groups finish.
GROUPED_PARTIAL_ENABLE=${GROUPED_PARTIAL_ENABLE:-False}
GROUPED_PARTIAL_READY_RATIO=${GROUPED_PARTIAL_READY_RATIO:-1.0}
GROUPED_PARTIAL_TARGET_PROMPT_GROUPS=${GROUPED_PARTIAL_TARGET_PROMPT_GROUPS:-${TRAIN_BATCH_SIZE}}
GROUPED_PARTIAL_MIN_PROMPT_GROUPS=${GROUPED_PARTIAL_MIN_PROMPT_GROUPS:-}
GROUPED_PARTIAL_GATE_POLL_INTERVAL=${GROUPED_PARTIAL_GATE_POLL_INTERVAL:-0.25}
GROUPED_PARTIAL_READY_GRACE_WAIT_SECONDS=${GROUPED_PARTIAL_READY_GRACE_WAIT_SECONDS:-5}
GROUPED_PARTIAL_MAX_OFF_POLICY_STEPS=${GROUPED_PARTIAL_MAX_OFF_POLICY_STEPS:-5}

# ---- agent loop (agents/agents.yaml reads the remaining AGENT_* / penalty vars) ----
AGENT_LOOP_WORKERS=${AGENT_LOOP_WORKERS:-32}
AGENT_MAX_CONCURRENT_ROLLOUTS=${AGENT_MAX_CONCURRENT_ROLLOUTS:-512}

if is_true "${ROLLOUT_CORRECTION_BYPASS_MODE}" && ! is_true "${ROLLOUT_CALCULATE_LOG_PROBS}"; then
    echo "ROLLOUT_CORRECTION_BYPASS_MODE=True requires ROLLOUT_CALCULATE_LOG_PROBS=True." >&2
    exit 7
fi
if is_true "${GROUPED_PARTIAL_ENABLE}" && [[ "${TRAINER_MODULE}" != "verl.trainer.main_ppo_sync" ]]; then
    echo "GROUPED_PARTIAL_ENABLE=True requires TRAINER_MODULE=verl.trainer.main_ppo_sync." >&2
    exit 6
fi

export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export WANDB_MODE=${WANDB_MODE:-online}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}
mkdir -p "${OUTPUT_DIR}"

ARGS=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.rollout_correction.bypass_mode=${ROLLOUT_CORRECTION_BYPASS_MODE}
    algorithm.rollout_correction.loss_type=${ROLLOUT_CORRECTION_LOSS_TYPE}
    algorithm.rollout_correction.rollout_is=${ROLLOUT_CORRECTION_IS}
    algorithm.rollout_correction.rollout_is_threshold=${ROLLOUT_CORRECTION_IS_THRESHOLD}
    algorithm.rollout_correction.rollout_is_batch_normalize=${ROLLOUT_CORRECTION_IS_BATCH_NORMALIZE}
    algorithm.rollout_correction.pre_update_metric_mode=${ROLLOUT_CORRECTION_PRE_UPDATE_METRIC_MODE}

    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.train_max_samples=${TRAIN_MAX_SAMPLES}
    data.val_max_samples=${VAL_MAX_SAMPLES}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=True
    data.seed=${DATA_SEED}
    data.dataloader_num_workers=0
    data.return_raw_chat=True
    data.return_multi_modal_inputs=False
    +data.apply_chat_template_kwargs.enable_thinking=$(is_true "${DISABLE_THINKING}" && echo False || echo True)

    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}
    +actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPLEMENTATION}
    actor_rollout_ref.model.enable_activation_offload=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_fused_kernels=${USE_FUSED_KERNELS}
    actor_rollout_ref.model.fused_kernel_options.impl_backend=${FUSED_KERNELS_BACKEND}
    actor_rollout_ref.model.lora_rank=${LORA_RANK}
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
    actor_rollout_ref.model.target_modules=${LORA_TARGET_MODULES}
    actor_rollout_ref.model.lora.merge=${LORA_MERGE}
    actor_rollout_ref.model.mtp.enable=${MTP_ENABLE}
    actor_rollout_ref.model.mtp.enable_rollout=${MTP_ENABLE_ROLLOUT}
    actor_rollout_ref.model.mtp.enable_train=${MTP_ENABLE_TRAIN}
    actor_rollout_ref.model.mtp.method=mtp
    actor_rollout_ref.model.mtp.num_speculative_tokens=${MTP_NUM_SPECULATIVE_TOKENS}

    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.calculate_entropy=${ACTOR_CALCULATE_ENTROPY}
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.policy_loss.loss_mode=${ACTOR_POLICY_LOSS_MODE}
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE}
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
    actor_rollout_ref.actor.fsdp_config.offload_policy=False
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}

    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.fsdp_config.strategy=fsdp2
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
    actor_rollout_ref.ref.fsdp_config.offload_policy=False
    actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD}

    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=${ROLLOUT_ENABLE_PREFIX_CACHING}
    actor_rollout_ref.rollout.disable_log_stats=${ROLLOUT_DISABLE_LOG_STATS}
    actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER}
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.load_format=${ROLLOUT_LOAD_FORMAT}
    actor_rollout_ref.rollout.layered_summon=${ROLLOUT_LAYERED_SUMMON}
    actor_rollout_ref.rollout.calculate_log_probs=${ROLLOUT_CALCULATE_LOG_PROBS}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable
    actor_rollout_ref.rollout.agent.default_agent_loop=agent_hub
    actor_rollout_ref.rollout.agent.agent_loop_config_path=agents/agents.yaml
    actor_rollout_ref.rollout.agent.num_workers=${AGENT_LOOP_WORKERS}
    actor_rollout_ref.rollout.agent.max_concurrent_rollouts=${AGENT_MAX_CONCURRENT_ROLLOUTS}

    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.balance_batch=True
    trainer.resume_mode=${TRAINER_RESUME_MODE}
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.log_val_generations=${LOG_VAL_GENERATIONS}
)
if [ -n "${ROLLOUT_GDN_PREFILL_BACKEND}" ]; then
    ARGS+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=${ROLLOUT_GDN_PREFILL_BACKEND})
fi
if [ -n "${ROLLOUT_ATTENTION_BACKEND}" ]; then
    ARGS+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=${ROLLOUT_ATTENTION_BACKEND})
fi
if is_true "${MTP_ENABLE_ROLLOUT}"; then
    export VERL_VLLM_SPEC_STATS_PATH=${VERL_VLLM_SPEC_STATS_PATH:-${OUTPUT_DIR%/}/${EXPERIMENT_NAME}_spec_stats.jsonl}
    if [ -n "${MTP_DRAFT_ATTENTION_BACKEND}" ]; then
        ARGS+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.attention_backend=${MTP_DRAFT_ATTENTION_BACKEND})
    fi
fi
if is_true "${GROUPED_PARTIAL_ENABLE}"; then
    ARGS+=(
        +trainer.grouped_partial_rollout.enable=True
        +trainer.grouped_partial_rollout.ready_ratio=${GROUPED_PARTIAL_READY_RATIO}
        +trainer.grouped_partial_rollout.target_prompt_groups=${GROUPED_PARTIAL_TARGET_PROMPT_GROUPS}
        +trainer.grouped_partial_rollout.gate_poll_interval=${GROUPED_PARTIAL_GATE_POLL_INTERVAL}
        +trainer.grouped_partial_rollout.ready_grace_wait_seconds=${GROUPED_PARTIAL_READY_GRACE_WAIT_SECONDS}
        +trainer.grouped_partial_rollout.max_off_policy_steps=${GROUPED_PARTIAL_MAX_OFF_POLICY_STEPS}
    )
    if [ -n "${GROUPED_PARTIAL_MIN_PROMPT_GROUPS}" ]; then
        ARGS+=(+trainer.grouped_partial_rollout.min_prompt_groups=${GROUPED_PARTIAL_MIN_PROMPT_GROUPS})
    fi
fi
if [ -n "${TRAINER_RESUME_FROM_PATH}" ]; then
    ARGS+=(trainer.resume_from_path="${TRAINER_RESUME_FROM_PATH}")
fi
if [ -n "${TOTAL_EPOCHS}" ]; then
    ARGS+=(trainer.total_epochs=${TOTAL_EPOCHS})
fi

echo "[train.sh] ${TRAINER_MODULE} experiment=${EXPERIMENT_NAME}"
echo "[train.sh] batch=${TRAIN_BATCH_SIZE}x${ROLLOUT_N} mini_batch=${PPO_MINI_BATCH_SIZE} epochs=${PPO_EPOCHS} lr=${ACTOR_LR}"
echo "[train.sh] lora rank=${LORA_RANK} alpha=${LORA_ALPHA} prompt=${MAX_PROMPT_LENGTH} response=${MAX_RESPONSE_LENGTH}"
echo "[train.sh] fsdp=${FSDP_SIZE} sp=${SP_SIZE} tp=${ROLLOUT_TP} prefix_cache=${ROLLOUT_ENABLE_PREFIX_CACHING} mtp=${MTP_ENABLE_ROLLOUT}"

exec python3 -m "${TRAINER_MODULE}" "${ARGS[@]}" "$@"
