# Agent loops

verl's agent-loop rollout calls `AgentHubLoop.run` (`agents/base_agent.py`)
once per sample. The hub builds a `TaskContext` and dispatches on the sample's
`data_source` to a plain async function:

```python
async def agent_loop(data: dict, context: TaskContext) -> AgentLoopOutput: ...
```

```python
AGENT_LOOPS = {
    "sandbox_agent": "agents.sandbox_agent.agent:agent_loop",
    "math": "agents.math.agent:agent_loop",
}
```

Unknown data sources fall back to `default_agent` in `agents/agents.yaml`.
Every other key in that YAML becomes `context.agent_options`. Most keys read an
environment variable, so recipes configure agents through env.

## Dataset rows

Rows are standard verl RL parquet rows. With `data.return_raw_chat=True`, the
agent receives them as `data`:

```python
{
    "data_source": "sandbox_agent",
    "prompt": [{"role": "user", "content": "..."}],      # optional system turn first
    "reward_model": {"style": "rule", "ground_truth": "72"},
    "extra_info": {"task_id": "suite/task_17", "reward_mode": "sandbox", ...},
}
```

Keep `extra_info` keys identical across rows (parquet structs). Any agent
option placed in `extra_info` overrides `agents.yaml` for that sample (for
example `max_turns` or `tool_timeout`). `extra_info.tools` restricts the tool
list.

## The `Agent` recorder (`agents/core/agent.py`)

`Agent(context, chat)` keeps the token-level trajectory that is trained:

- `await agent.step(max_new_tokens=None)` samples one assistant turn through
  verl's rollout server. Sampled tokens are trainable (`response_mask=1`) and
  carry vLLM log-probs. It returns the decoded text, or `None` once the
  response budget is spent.
- `agent.append({"role": "user", "content": obs})` appends an observation as
  non-trainable tokens. Only the token delta of the chat template is appended,
  so the context stays an exact prefix of the next request. Prefix caching
  depends on this.
- `agent.add_process_reward_to_last_action(r)` adds `r` to the
  `process_advantages` of the last sampled turn's tokens. The trainer adds
  these to the GRPO advantages (`main_ppo_sync._add_process_advantages`).
- `await agent.get_agent_output(reward, extra_info=metrics)` packs prompt,
  response, masks, log-probs and process advantages into an `AgentLoopOutput`.
  Numeric `extra_info` entries with a `/` in the key are averaged into the
  `aa/<key>` training metrics, and binary `.../is_valid` keys also produce
  `pass@N` during validation.
- `agent.react(run_action)` is a minimal step/observe loop for simple agents.

## Sandbox agent (`agents/sandbox_agent/`)

The sandbox agent is a generic tool-use loop. Each episode runs as follows:

1. Open a sandbox session and a trial for `extra_info.task_id`.
2. Build the system prompt from the tool schemas (`prompts.py`). The tools are
   bash, execute_python, read/write/str_replace file and return_answer, in the
   XML `qwen3_coder` call format.
3. Repeat: generate a turn, parse its calls (`agents/core/tool_calls.py`),
   execute up to `max_calls_per_turn`, and append the observations. Stop on
   `return_answer`, `max_turns`, a full context, or a fatal sandbox error.
4. Score the episode:
   - **`reward_mode=match`**: compare the answer to `reward_model.ground_truth`
     (normalized exact match, then Math-Verify).
   - **`reward_mode=sandbox`**: call the task's server-side reward
     (`verify.py` or a hidden-test command).
5. Optionally write one JSON line per rollout to `TRAJECTORY_TRACE_PATH`.

Process rewards (`PROCESS_REWARDS_ENABLED=True`, training only) have two
parts:

- **Per-turn penalty**: a capped penalty on the turn's tokens when the turn had
  a parse error, a tool error or a tool timeout (`PARSE_ERROR_PENALTY`,
  `TOOL_ERROR_PENALTY`, `TOOL_TIMEOUT_PENALTY`, `ACTION_ERROR_PENALTY_CAP`).
  Each error type counts once per turn, so emitting fewer calls per turn is not
  rewarded.
- **Context penalty**: a sequence-level penalty that ramps linearly to
  `CONTEXT_PENALTY_MAX` inside the last `CONTEXT_PENALTY_BUFFER_TOKENS` of the
  response budget, and applies in full when the context fills up.

Logged metrics (`aa/agent/*`):

| Group | Metrics |
|---|---|
| Reward | `reward` (training reward), `base_reward_score`, `is_valid`, `answered`, `context_penalty` |
| Activity | `turns`, `tool_calls`, `tool_seconds`, `min_remaining_tokens`, `duration_sec` |
| Errors and shaping | `parse_errors`, `tool_errors`, `tool_timeouts`, `process_reward_sum`, `process_reward_actions` |
| Termination | `terminated_{return_answer,context_full,max_turns,llm_response_failed,fatal_error}` |

## Writing a new agent

```python
from agents.core.agent import Agent

async def agent_loop(data, context):
    agent = Agent(context, list(data["raw_prompt"]))
    for _ in range(8):
        text = await agent.step()
        if text is None or "FINAL:" in text:
            break
        agent.append({"role": "user", "content": my_env.observe(text)})
    reward = my_env.score(text)
    return await agent.get_agent_output(reward, extra_info={"myenv/score": reward})
```

Register it in `AGENT_LOOPS` and set `data_source` accordingly in your
dataset. Test it without GPUs by giving `TaskContext` a real tokenizer and a
scripted `llm_client` (see `tests/agents/test_sandbox_agent_e2e.py`).
