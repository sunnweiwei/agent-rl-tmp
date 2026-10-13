# Sandbox server

`sandbox_server/` is a FastAPI service that runs agent actions in isolation and
computes trusted rewards. Rollout workers only need `SANDBOX_MANAGER_URL` (and
`SANDBOX_API_TOKEN`), and talk to it through `agents/sandbox_agent/client.py`.

```
rollout worker ──HTTP──▶ SandboxManager ──▶ ActionDispatcher ──▶ nsjail jail / docker exec
                           sessions          file tools → workspace
                           trials            bash / python → isolation backend
                           rewards           verify.py (trusted) / command (isolated)
```

## Concepts

- **Profile.** Picks a backend plus its limits. The `nsjail` profile runs a
  fresh jail per action over the trial workspace, using the rootfs built into
  `docker/sandbox/Dockerfile`. The `docker` profile runs one container per
  trial, with the image from `task.yaml`.
- **Session.** One per rollout, or one shared per job. It is bound to a profile.
- **Trial.** One per episode. It owns a workspace directory mounted at
  `/workspace`, which is also the working directory. Trials are leased: every
  action or heartbeat extends the lease, and expired trials are reaped.
- **Action.** A tool call. Actions within one trial run one at a time; across
  trials they are bounded by `max_running_actions` and an optional per-tool
  limit.

## Tools

| Tool | Args | Result |
|---|---|---|
| `bash` | `command` | stdout (plus stderr); non-zero exit is an error carrying the output |
| `execute_python` | `code` | stdout of a fresh interpreter; non-zero exit is an error |
| `read_file` | `path` | file text (path confined to the workspace, no symlinks) |
| `write_file` | `path`, `content` | `{"ok": true, "path": ...}` |
| `str_replace_file` | `path`, `old_str`, `new_str` | replaces the first occurrence |
| `return_answer` | `answer` | sentinel; the agent ends the episode |

## Task directories

The client sends only `task_id`, a path relative to `SANDBOX_TASK_ROOT`. An
empty `task_id` creates a scratch trial with an empty workspace and no
server-side reward.

```
<task_root>/<task_id>/
  task.yaml     optional
  workspace/    copied into the trial workspace at creation
  tests/        hidden; copied to .eval/tests only when the reward runs
  verify.py     hidden; trusted answer check
```

```yaml
# task.yaml
image: python:3.12-slim          # docker profile only
setup_commands:                  # run in the trial after creation; failure aborts the trial
  - pip install -e .
reward:
  type: command                  # or "verify" (default when verify.py exists)
  command: python -m pytest -q .eval/tests
  timeout: 600
```

- `verify` imports `verify.py` in the server process and calls
  `verify(answer)` or `verify(answer, workspace_path)`. It may return a bool, a
  float, `(score, reason)`, or `{"score", "passed", "reason"}`.
- `command` runs inside the trial with the agent's isolation. Exit code 0 gives
  reward 1.0.

## HTTP API

| Method | Path | Body | Notes |
|---|---|---|---|
| GET | `/v1/health` | | no auth |
| GET | `/v1/capacity` | | live/running/queued counters per profile and tool |
| POST | `/v1/sessions` | `{"profile"}` | |
| DELETE | `/v1/sessions/{id}` | | destroys its trials |
| POST | `/v1/sessions/{id}/trials` | `{"task_id", "lease_seconds"}` | 429 when capacity is exhausted |
| POST | `/v1/trials/{id}/actions` | `{"tool", "args", "timeout"}` | `{"success", "result", "error", ...}` |
| POST | `/v1/trials/{id}/heartbeat` | | extends the lease |
| POST | `/v1/trials/{id}/reward` | `{"answer", "timeout"}` | 501 if the task has no reward |
| DELETE | `/v1/trials/{id}` | | |

When `SANDBOX_API_TOKEN` is set on the server, every endpoint except
`/v1/health` requires `Authorization: Bearer <token>`.

## Configuration (env)

| Variable | Default | Meaning |
|---|---|---|
| `SANDBOX_TASK_ROOT` | — | task directories |
| `SANDBOX_WORKSPACE_ROOT` | `/tmp/agent-sandbox` | trial workspaces (put on fast local disk) |
| `SANDBOX_ENABLED_PROFILES` | all | e.g. `nsjail` |
| `SANDBOX_MAX_LIVE_TRIALS` | 1024 | trials with workspace state (most are idle during inference) |
| `SANDBOX_MAX_RUNNING_ACTIONS` | 128 | concurrent actions; main CPU knob, keep ≤ vCPUs − 4 |
| `SANDBOX_MAX_QUEUED_ACTIONS` | 50000 | reject beyond this queue depth |
| `SANDBOX_TOOL_MAX_RUNNING_ACTIONS` | — | per-tool limits, `bash:512,execute_python:512` |
| `SANDBOX_LEASE_SECONDS` | 900 | idle trial reap timeout |
| `SANDBOX_REWARD_TIMEOUT_SECONDS` | 600 | default reward deadline |
| `SANDBOX_MEMORY_SOFT_LIMIT` | 192g | reject new work above this RSS |
| `SANDBOX_DOCKER_IMAGE`, `_MEM_LIMIT`, `_CPU_LIMIT`, `_PIDS_LIMIT`, `_NETWORK` | `python:3.12-slim`, 8g, 2, 512, none | docker profile |
| `SANDBOX_ALLOW_UNSAFE_SUBPROCESS` | 0 | dev/test only: run actions as plain host subprocesses |

## Performance notes

- An nsjail action costs about 1.5 s of Python cold start, so the server
  sustains tens of actions per second per 128 slots. In real rollouts, LLM
  inference dominates wall time.
- With the docker profile, containerd caps parallel container starts at about
  10; beyond 64 concurrent creates, iptables locking serializes them. Batch
  trial creation if you need more.
- Watch `/v1/capacity` for growing `queued_actions` during rollout.

## Local development

```bash
SANDBOX_ALLOW_UNSAFE_SUBPROCESS=1 SANDBOX_ENABLED_PROFILES=nsjail SANDBOX_TASK_ROOT=~/data/sandbox_tasks \
    python -m uvicorn sandbox_server.server:app --port 8084
```
