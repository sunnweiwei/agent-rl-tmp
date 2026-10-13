# agent-rl

Multi-turn agent RL for ~27B models on a single 8-GPU node, built on a fork of
[verl](https://github.com/verl-project/verl). The policy runs a tool-use loop
against an isolated sandbox server, and GRPO trains it on outcome rewards plus
optional turn-level process penalties.

The default recipes train **Qwen3.6-27B** (LoRA or full-parameter) with a 16K
prompt / 48K response budget on 8x B200 or 8x H100, with these components:

- **Rollout**: vLLM with prefix caching, LoRA hot-swap, and the FLA fused
  GatedDeltaNet prefill path for Qwen3.5/3.6.
- **Training**: FSDP2 with packed sequences, fused LM-head kernels and dynamic
  batching.
- **Agent loop**: function-style agents (`agents/`) that call the model
  token-in/token-out, so the trained tokens are exactly the sampled ones.
- **Sandbox**: a FastAPI server (`sandbox_server/`) that runs agent actions in
  nsjail jails or per-trial Docker containers and computes trusted rewards.

## Layout

```
agents/               agent loops (verl AgentLoop adapter + function-style agents)
  base_agent.py       AgentHubLoop: routes each sample by data_source
  agents.yaml         agent options (env-overridable), loaded by the trainer
  core/               Agent trajectory recorder, XML tool-call parser, process rewards
  sandbox_agent/      generic bash/python/file tool-use agent backed by the sandbox
  math/               single-turn math agent (no tools)
sandbox_server/       sandbox HTTP server: sessions, trials, actions, rewards
recipes/              validated launch recipes (env presets)
scripts/
  launch_docker.sh    host side: start the training container
  train.sh            container side: build Hydra overrides, run the trainer
  monitor_training.sh crash/stall/KL/val monitor for a running container
  diagnostics/        B200 kernel, vLLM sleep/wake, LoRA and MTP correctness probes
docker/
  b200-training/      B200 training image (vLLM patches for FLA 0.5.1 GDN, RL sleep/wake, LoRA)
  h100-sampling/      H100 image variant
  sandbox/            sandbox server image (nsjail + rootfs)
configs/vllm/         tuned vLLM LoRA kernel configs (required on B200)
examples/             dataset builders
verl/                 the verl fork (changes listed in docs/verl_changes.md)
docs/                 training runbook, sandbox, agents, verl changes
```

## Quickstart

### 1. Start the sandbox

The sandbox can run on the training node or on a separate CPU node. Every
task is a directory under `SANDBOX_TASK_ROOT` (see [docs/sandbox.md](docs/sandbox.md)).

```bash
python examples/prepare_sandbox_data.py --output_dir ~/data/sandbox_agent --task_root ~/data/sandbox_tasks

export SANDBOX_API_TOKEN=$(openssl rand -hex 32)
SANDBOX_TASK_ROOT=~/data/sandbox_tasks docker compose -f docker/sandbox/docker-compose.yml up -d --build
curl -fsS http://127.0.0.1:8084/v1/health
```

### 2. Write the env files

Keep secrets outside the repository. `scripts/launch_docker.sh` reads both
files from the repository's parent directory by default.

```bash
cat > ../.wandb_env <<EOF
WANDB_API_KEY=<key>
WANDB_PROJECT=agent-rl
EOF
cat > ../.sandbox_env <<EOF
SANDBOX_MANAGER_URL=http://127.0.0.1:8084
SANDBOX_API_TOKEN=${SANDBOX_API_TOKEN}
EOF
chmod 600 ../.wandb_env ../.sandbox_env
```

### 3. Train

```bash
MODEL_PATH=/models/Qwen3.6-27B \
DATA_DIR=~/data/sandbox_agent \
OUTPUT_DIR=/outputs/qwen36-27b-lora \
bash recipes/qwen36_27b_lora_b200.sh
```

The launcher prints the container name and writes it to
`${OUTPUT_DIR}/container_name`. Per-rollout trajectories are appended to
`${OUTPUT_DIR}/trajectories_<run>.jsonl`. Any extra arguments are passed
through to the trainer as Hydra overrides.

```bash
docker logs -f "$(cat /outputs/qwen36-27b-lora/container_name)"
OUTPUT_DIR=/outputs/qwen36-27b-lora bash scripts/monitor_training.sh
```

## Recipes

| Recipe | Hardware | Training | Batch | Parallelism | Context |
|---|---|---|---|---|---|
| `qwen36_27b_lora_b200.sh` | 8x B200 | LoRA r32/α64, lr 5e-5 | 16 prompts x 16, 4 updates/step | FSDP8, SP1, TP1 | 16K + 48K |
| `qwen36_27b_full_b200.sh` | 8x B200 | full, lr 1e-6 | 16 prompts x 16, 4 updates/step | FSDP8, SP1, TP1 | 16K + 48K |
| `qwen36_27b_lora_h100.sh` | 8x H100 | LoRA r32/α32, lr 5e-5 | 32 prompts x 8 | FSDP8, SP8, TP8 | 16K + 48K |

All recipes use GRPO without KL loss, `seq-mean-token-mean` loss aggregation,
asymmetric clipping (0.2 / 0.28), and vLLM behavior log-probs (rollout
correction in bypass mode). Thinking is disabled, prefix caching is on, and MTP
is off. Every value is an environment variable defaulted in `scripts/train.sh`;
recipes only set what differs. The B200 recipes reproduce the composed trainer
config of the original validated runs exactly.

See [docs/training.md](docs/training.md) for preflight checks, healthy early
signals, monitoring, resume, and known failure modes.

## Adding your own environment

You have two options:

- **New tasks for the sandbox agent.** Write task directories (`workspace/`,
  `verify.py` or a `tests/` + `command` reward) and dataset rows with
  `data_source: sandbox_agent` and `extra_info.task_id`. No code changes.
- **A new agent loop.** Write
  `async def agent_loop(data, context) -> AgentLoopOutput` using
  `agents.core.agent.Agent`, then register it in `AGENT_LOOPS` in
  `agents/base_agent.py` under the `data_source` it should handle.

[docs/agents.md](docs/agents.md) covers the dataset row format, the `Agent`
API, process rewards and logged metrics.

## Tests

```bash
pytest tests/agents tests/sandbox_server/unit tests/trainer tests/experimental/agent_loop tests/models -q
# with a local Qwen tokenizer, also run the scripted agent-loop end-to-end test:
AGENT_TEST_TOKENIZER=/models/Qwen3.6-27B pytest tests/agents/test_sandbox_agent_e2e.py -q
```

`tests/kernels/b200_*` and `scripts/diagnostics/` must run inside the B200
training image on GPUs.

## License

Apache 2.0, inherited from verl (see `LICENSE` and `Notice.txt`).
