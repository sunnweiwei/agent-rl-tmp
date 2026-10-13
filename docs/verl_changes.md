# Changes to verl

`verl/` is forked from upstream
[verl-project/verl@a1f6e71](https://github.com/verl-project/verl/commit/a1f6e7197ed8c7798278b8a26e5059e0bb93b39e)
(2026-06-09, version `0.9.0.dev`). To list the exact diff:

```bash
git clone https://github.com/verl-project/verl /tmp/verl && cd /tmp/verl && git checkout a1f6e71
diff -ru /tmp/verl/verl <repo>/verl
```

## Trainer (`verl/trainer/main_ppo_sync.py`)

- **Grouped partial rollout** (`trainer.grouped_partial_rollout.*`). A rollout
  gate plus a resumable LLM client let training start once enough prompt
  groups have finished. Unfinished groups are replayed with a bounded
  off-policy staleness.
- **Process advantages.** Token-level `process_advantages` from the agent loop
  are added to the GRPO advantages and returns. Logged as `process_reward/*`.
- **Curriculum weighting** (`algorithm.curriculum.*`). Per-task EMA success
  rates down-weight tasks outside a success band. It reads
  `agent/base_reward_score` by default.
- **Diagnostics and control.** Pre-update rollout-vs-actor log-prob metrics
  (`algorithm.rollout_correction.pre_update_metric_mode`: `off`,
  `first_minibatch` or `full_batch`), vLLM speculative-decoding stats, and
  early stopping on a validation metric.

## Algorithms and metrics

- **`core_algos.py`**: the `foldgrpo` advantage estimator, which is GRPO that
  counts each rollout (`gen_uid`) once when an agent emits several trainable
  sequences per rollout. `ray_trainer.py` and `padding_utils.py` carry
  `gen_uid` through.
- **`metric_utils.py`**: per-rollout de-duplication for multi-output agents,
  `train/informative_ratio` (the share of groups with non-constant reward),
  `aa/<key>` averages of every numeric `reward_extra_info` entry, and exact
  `pass@N` for binary `*/is_valid` keys.
- **`rollout_corr_helper.py`**: compact and detailed off-policy metric helpers.
- **`utils/metric/utils.py`**: DP metric aggregation that tolerates different
  micro-batch counts per rank under dynamic batching.
- **`utils/tracking.py`**: W&B logs with `commit=True`, so the dashboard is not
  one step behind.

## Agent loop (`verl/experimental/agent_loop/`)

- **Process advantages.** `AgentLoopOutput.process_advantages` is padded into
  the batch.
- **Multi-output rollouts.** An agent loop may return a list of outputs, and
  the non-tensor inputs are expanded to match.
- **Concurrency limit.** `rollout.agent.max_concurrent_rollouts` sets a global
  limit split across agent-loop workers.
- **Tracking.** The full weight-version range (`min/max_global_steps`), LoRA
  applied status, and partial-rollout segment counts are tracked across turns.

## Models and engines

- **Qwen3.5/3.6** (`models/transformers/qwen3_5.py`, `monkey_patch.py`,
  `tiled_mlp.py`): packed varlen forward through the FLA fused GatedDeltaNet
  kernels with remove-padding. The offset RMSNorm is kept, because the Liger
  replacement changes full-FT gradients. Also adds tiled MLP support.
- **FSDP engine** (`workers/engine/fsdp/transformer_impl.py`) has two fixes:
  - It casts LoRA adapter params to the base bf16 dtype. Without the cast,
    backward fails with a "bf16 grad into fp32 param" error under FSDP2 mixed
    precision.
  - It detaches the stored `model_output` after `loss.backward()`. Otherwise
    every micro-batch's autograd graph is retained, which caused an OOM in the
    27B LoRA actor update.
- **Optimizer offload** (`workers/engine/base.py`, `engine_workers.py`): the
  offloaded optimizer state is loaded only right before `optimizer_step`, after
  activations are freed. Also adds per-PPO-epoch metric aliases.
- **Fused LM-head backward** (`utils/experimental/torch_functional.py`): the
  vocabulary gradient is accumulated in place instead of materializing another
  `[vocab, hidden]` tensor.
- **Losses** (`workers/utils/losses.py`): accept already-padded log-prob
  outputs, return a zero loss for fully masked micro-batches, and emit the
  first-mini-batch pre-update metrics.

## vLLM rollout (`workers/rollout/vllm_rollout/`)

- LoRA weight sync with optional fingerprinting (`VERL_LORA_SYNC_DEBUG`,
  implemented in `utils/lora_sync_debug.py`). Strict weight-version checks are
  enabled with `VERL_ROLLOUT_WEIGHT_VERSION_STRICT`.
- Prefix-cache statistics (`num_cached_tokens`) are exposed per request, along
  with MTP speculative-decoding stats.
- The MTP drafter is synced and gated (`VERL_SYNC_MTP_DRAFTER`), with an
  online-quantization check.

## Config

- `rollout.yaml`: adds `agent.max_concurrent_rollouts`.
- `algorithm/rollout_correction.yaml`: adds `pre_update_metric_mode`.
- The `_generated_*` trainer configs are regenerated accordingly.

The vLLM engine itself is patched in the training images. See
`docker/b200-training/*.patch`: FLA 0.5.1 fused GDN prefill, RL sleep/wake
synchronization with a partial-wake scheduler gate, and LoRA int64 row
indices. `scripts/preflight_image.sh` checks that these patches are present.
