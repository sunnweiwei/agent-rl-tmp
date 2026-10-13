# Training runbook

The launch path has three layers. Each one only reads environment variables:

```
recipes/<recipe>.sh        env presets for one validated setup
  └─ scripts/launch_docker.sh   host: mounts, GPUs, env forwarding, docker run
       └─ scripts/preflight_image.sh + scripts/train.sh   container: Hydra overrides → verl.trainer.main_ppo_sync
```

To change an experiment, export a variable before calling the recipe
(`ACTOR_LR=2e-5 bash recipes/...`) or append Hydra overrides as arguments. You
should not need to edit `train.sh`.

## Invariants of the validated 27B path

- **B200 kernel path.** Keep the packed and fused path: FLA fused GDN
  (`FLA_TILELANG=0`), fused kernels, remove-padding, Triton attention, and the
  patched vLLM sleep/wake. The B200 recipes hard-require these, and
  `preflight_image.sh` fails the container when the image lacks them.
- **Sequence parallelism on B200.** Keep `SP_SIZE=1` there. Ulysses SP=8 hit a
  misaligned-address failure with FLA context parallelism on B200; the packed
  SP=1 path is covered by the B200 diagnostics.
- **Dynamic batching.** `PPO_MAX_TOKEN_LEN_PER_GPU` must fit the longest single
  sequence (prompt + response, 64K in the recipes).
- **Prefix cache and MTP.** Keep prefix caching on and MTP off. Multi-turn
  agents rely on intra-session reuse: turn t+1 extends turn t's tokens exactly,
  and the agent loop keeps that true. MTP was numerically correct on B200, but
  it reduced cache reuse and throughput.
- **Offloads.** Use stage-boundary offloads only (`ACTOR_PARAM_OFFLOAD`,
  `ACTOR_OPTIMIZER_OFFLOAD`, `REF_PARAM_OFFLOAD`). `train.sh` pins FSDP runtime
  `offload_policy=False`, so forward, backward and the optimizer step stay on
  the GPU.
- **Validation data.** Validate on a held-out set that does not overlap the
  train set. If train tasks were selected by sampling a frozen model, check
  `(task_id)` overlap before launching.
- **Secrets.** Keep them in env files outside the repository (`../.wandb_env`,
  `../.sandbox_env`).

## Preflight

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader   # all 8 idle
test -f "${MODEL_PATH}/config.json"
curl -fsS "${SANDBOX_MANAGER_URL}/v1/health"
curl -fsS -H "Authorization: Bearer ${SANDBOX_API_TOKEN}" "${SANDBOX_MANAGER_URL}/v1/capacity"
docker pull "${IMAGE}"
```

If `/v1/capacity` returns 401, the URL and token do not match. A non-local
sandbox URL requires an explicit `SANDBOX_API_TOKEN`; the launcher refuses to
start without one.

## Healthy early signals

The startup log should contain the following:

- **Preflight lines** (B200 recipes):
  `[preflight] FLA fused GDN backend enabled through Triton`,
  `[preflight] vLLM forward_native uses FLA 0.5.1 fused ... GDN prefill`, and
  `[preflight] vLLM RL sleep/wake synchronization and partial-wake scheduler gate enabled`.
- **Model**: `PeftModelForCausalLM contains 27.61B parameters` for LoRA runs,
  and a LoRA adapter dtype cast to bf16.
- **Rollouts**: no 401 or 5xx from the sandbox, and the trajectories JSONL
  starts growing during the first rollout.
- **Prefix cache**: `rollout/prefix_cache_hit_rate` climbs above 0.9 after cold
  start (0.93–0.94 on the reference B200 run).
- **Actor update**: the first update completes without CUDA OOM. On B200 with
  the 0.86–0.90 memory setting, each GPU holds about 3.8M KV-cache tokens.

A low reward in the first steps is not a failure. Judge early health by sandbox
connectivity, trajectory growth, GPU utilization and a completed actor update.
A clean first step is necessary, but it does not prove the run is stable.

## Monitoring

```bash
OUTPUT_DIR=<run output dir> INTERVAL_SECONDS=300 bash scripts/monitor_training.sh
python scripts/summarize_wandb_rl_run.py <entity/project/run_id>
```

Watch these signals:

- **Infrastructure**: container state and OOM-kill, sandbox 5xx or session
  failures, GPU Xid/ECC errors.
- **Rollout**: prefix-cache hit rate and stale-weight requests
  (`rollout/weight_step_stale_requests`).
- **Optimization**: validation reward (`val-core/<data_source>/reward/mean@N`),
  pre-update KL (`actor/pre_update/kl`), clip fraction and response length.
- **Agent behavior** (`aa/agent/*`): `terminated_*` reasons, `parse_errors`,
  `tool_errors`, `tool_timeouts`, `process_reward_sum`, `context_penalty`.
- **Trajectories**: the JSONL holds full messages, tool calls and observations
  for offline analysis.

## Resume

Resume only from a complete checkpoint, with the same data and recipe. Set
`SAVE_FREQ` at launch, since the recipes default to no checkpoints. Then
relaunch with:

```bash
TRAINER_RESUME_MODE=resume_path TRAINER_RESUME_FROM_PATH=<ckpt dir> bash recipes/<recipe>.sh
```

## Known failure modes

| Symptom | Cause / fix |
|---|---|
| `SANDBOX_API_TOKEN must be set` | Non-local sandbox URL without a token. Export it or put it in `../.sandbox_env`. |
| `401 Unauthorized` in trajectories | Token does not match the server. Re-test `/v1/capacity` with the same token. |
| `bf16 grad into fp32 param` during LoRA backward | The LoRA adapter dtype cast in `verl/workers/engine/fsdp/transformer_impl.py` is missing, or the container mounted a different repo. |
| CUDA OOM in actor update, growing per micro-batch | `model_output` is not detached after `loss.backward()` (same file), so every micro-batch's graph is retained. Also confirm `PPO_MAX_TOKEN_LEN_PER_GPU` and the batch shape match the recipe. |
| Preflight assertion fails | The image lacks a required vLLM patch. Use the recipe's image or rebuild from `docker/b200-training`. |
| `misaligned address` on B200 | `SP_SIZE>1` with FLA context parallelism. Use `SP_SIZE=1`. |
| Many `terminated_fatal_error` | Sandbox unreachable or out of capacity. Check `/v1/capacity` and the server's memory soft limit. |
| Prefix-cache hit rate collapses mid-run | Check that observations are deterministic and that the context is not rewritten between turns; compare per-turn `prefix_cache_*` extra fields across runs. |

## Exporting a checkpoint

```bash
python scripts/legacy_model_merger.py merge --backend fsdp \
    --local_dir <ckpt>/global_step_N/actor --target_dir <hf_out>
```
