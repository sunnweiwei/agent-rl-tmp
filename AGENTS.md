# Agent instructions

- Entry points: `recipes/*.sh` → `scripts/launch_docker.sh` (host) → `scripts/train.sh` (container).
  All configuration is env vars defaulted in `scripts/train.sh` and `agents/agents.yaml`.
- Agent loops live in `agents/`; register new ones in `AGENT_LOOPS` (`agents/base_agent.py`).
- The sandbox server (`sandbox_server/`) resolves tasks from its own task root; clients only send `task_id`.
- `verl/` is a fork of upstream verl. Record every change to it in `docs/verl_changes.md`.
- Do not change a recipe's kernel flags (FLA fused GDN, fused kernels, remove-padding, SP=1 on B200,
  prefix caching on, MTP off) without a controlled A/B run.
- Never commit secrets; they live in `../.wandb_env` and `../.sandbox_env`.
- CPU tests: `pytest tests/agents tests/sandbox_server/unit tests/trainer tests/experimental/agent_loop tests/models -q`.
