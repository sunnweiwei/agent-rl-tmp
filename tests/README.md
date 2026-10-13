# Tests

| Path | What | Needs |
|---|---|---|
| `agents/` | tool-call parser, agent config, sandbox server + client + agent-loop end-to-end | CPU; set `AGENT_TEST_TOKENIZER` for the agent-loop test |
| `sandbox_server/unit/` | manager, actions, tasks, rewards, isolation backends | CPU |
| `trainer/`, `experimental/agent_loop/`, `models/`, `utils/`, `workers/` | custom verl changes (advantages, rollout correction, metrics, grouped partial rollout, Qwen3.5 packed backend) | CPU |
| `special_distributed/` | Qwen3.5 linear attention under Ulysses SP | multiple GPUs |
| `kernels/b200_*` | FLA / vLLM GDN kernel diagnostics | B200 training image |
