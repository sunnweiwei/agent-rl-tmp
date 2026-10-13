"""Generic multi-turn tool-use agent backed by the sandbox server.

One rollout = one sandbox trial. The model emits XML function calls, the
sandbox executes them, observations are appended as user turns, and the
episode ends at ``return_answer``, the turn limit, or the context limit.

Dataset rows (see ``examples/prepare_sandbox_data.py``)::

    prompt:       [{"role": "user", "content": <task>}]  (optional system turn)
    data_source:  "sandbox_agent"
    reward_model: {"ground_truth": ...}                  (only for reward_mode=match)
    extra_info:   {"task_id": <dir under SANDBOX_TASK_ROOT, or "">,
                   "reward_mode": "sandbox" | "match",   (default: match if ground_truth)
                   ... any option below, overriding agents.yaml per sample}
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any

from agents.core.agent import Agent, TaskContext
from agents.core.process_rewards import apply_action_process_penalty, context_limit_penalty
from agents.core.tool_calls import parse_tool_calls
from agents.sandbox_agent.client import SandboxClient, SandboxError
from agents.sandbox_agent.prompts import TOOLS, build_system_prompt

DEFAULTS: dict[str, Any] = {
    "sandbox_url": "http://127.0.0.1:8084",
    "sandbox_profile": "nsjail",
    "max_turns": 100,
    "max_calls_per_turn": 10,
    "tool_timeout": 120.0,
    "tool_total_budget": 1800.0,
    "max_observation_chars": 16000,
    "max_turn_observation_chars": 48000,
    "lease_seconds": 7200.0,
    "heartbeat_interval": 60.0,
    "process_rewards_enabled": False,
    "parse_error_penalty": 0.25,
    "tool_error_penalty": 0.05,
    "tool_timeout_penalty": 0.05,
    "action_error_penalty_cap": 0.5,
    "context_penalty_buffer_tokens": 8192,
    "context_penalty_max": 0.0,
    "trajectory_trace_path": "",
}
TERMINATE_REASONS = ("return_answer", "context_full", "max_turns", "llm_response_failed", "fatal_error")
NO_CALL_OBSERVATION = (
    "No function call was found in your response. Every turn must contain at least one "
    "<function=...>...</function> block. Call return_answer when you are done."
)
_REQUIRED = {tool["function"]["name"]: tool["function"]["parameters"]["required"] for tool in TOOLS}


def resolve_config(agent_options: dict[str, Any], extra_info: dict[str, Any]) -> dict[str, Any]:
    """Per-sample ``extra_info`` > ``agents.yaml`` options > defaults, cast to the default's type."""
    config = {}
    for name, default in DEFAULTS.items():
        value = default
        for source in (agent_options, extra_info):
            if source.get(name) not in (None, ""):
                value = source[name]
        if isinstance(default, bool) and isinstance(value, str):
            value = value.strip().lower() in ("1", "true", "yes")
        config[name] = type(default)(value)
    return config


async def agent_loop(data: dict[str, Any], context: TaskContext):
    extra_info = dict(data.get("extra_info") or {})
    config = resolve_config(context.agent_options, extra_info)
    tool_names = set(extra_info.get("tools") or _REQUIRED)
    tools = [tool for tool in TOOLS if tool["function"]["name"] in tool_names | {"return_answer"}]
    tool_names = {tool["function"]["name"] for tool in tools}

    chat = [dict(turn) for turn in (data.get("raw_prompt") or data.get("prompt") or [])]
    system = build_system_prompt(tools, config["max_calls_per_turn"])
    if chat and chat[0].get("role") == "system":
        chat[0] = {"role": "system", "content": f"{system}\n\n{chat[0]['content']}"}
    else:
        chat.insert(0, {"role": "system", "content": system})
    agent = Agent(context, chat)

    state: dict[str, Any] = {
        "task_id": str(extra_info.get("task_id", "")),
        "final_answer": "",
        "terminate_reason": "max_turns",
        "error": "",
        "turns": 0,
        "tool_calls": 0,
        "parse_errors": 0,
        "tool_errors": 0,
        "tool_timeouts": 0,
        "tool_seconds": 0.0,
        "process_reward_sum": 0.0,
        "process_reward_actions": 0,
        "min_remaining_tokens": context.response_length,
        "trajectory": [],
    }
    start = time.time()
    client = SandboxClient(config["sandbox_url"])
    session = trial = heartbeat = None
    try:
        session = await client.open_session(config["sandbox_profile"])
        trial = await session.create_trial(task_id=state["task_id"], lease_seconds=config["lease_seconds"])
        heartbeat = asyncio.create_task(_heartbeat(trial, config["heartbeat_interval"]))
        await _run_turns(agent, trial, state, config, tool_names, shaping=context.is_train)
    except Exception as exc:
        state["terminate_reason"] = "fatal_error"
        state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if heartbeat is not None:
            heartbeat.cancel()

    base_reward, is_valid, reward_reason = await _compute_reward(data, extra_info, trial, state["final_answer"])
    await _cleanup(trial, session)

    if not agent.response_ids:
        # Keep the sample well-formed when the episode died before generation.
        agent.append({"role": "user", "content": f"Agent terminated: {state['terminate_reason']} {state['error']}"})

    remaining = context.response_length - len(agent.response_ids)
    state["min_remaining_tokens"] = min(state["min_remaining_tokens"], remaining)
    context_penalty = 0.0
    if config["process_rewards_enabled"] and context.is_train:
        context_penalty = context_limit_penalty(
            remaining,
            buffer_tokens=config["context_penalty_buffer_tokens"],
            max_penalty=config["context_penalty_max"],
            context_full=state["terminate_reason"] == "context_full",
        )
    reward = base_reward + context_penalty

    metrics = {
        "agent/reward": reward,
        "agent/base_reward_score": base_reward,
        "agent/is_valid": int(is_valid),
        "agent/answered": int(bool(state["final_answer"])),
        "agent/context_penalty": context_penalty,
        "agent/duration_sec": round(time.time() - start, 3),
    }
    for key in (
        "turns",
        "tool_calls",
        "parse_errors",
        "tool_errors",
        "tool_timeouts",
        "tool_seconds",
        "process_reward_sum",
        "process_reward_actions",
        "min_remaining_tokens",
    ):
        metrics[f"agent/{key}"] = state[key]
    for reason in TERMINATE_REASONS:
        metrics[f"agent/terminated_{reason}"] = int(state["terminate_reason"] == reason)
    _write_trace(config["trajectory_trace_path"], state, metrics, reward_reason, agent.chat)
    return await agent.get_agent_output(reward, extra_info=metrics)


async def _run_turns(agent: Agent, trial, state: dict, config: dict, tool_names: set[str], shaping: bool) -> None:
    for _ in range(config["max_turns"]):
        remaining = agent.context.response_length - len(agent.response_ids)
        state["min_remaining_tokens"] = min(state["min_remaining_tokens"], remaining)
        if remaining < 10:
            state["terminate_reason"] = "context_full"
            return
        response = await agent.step()
        if response is None:
            remaining = agent.context.response_length - len(agent.response_ids)
            state["terminate_reason"] = "context_full" if remaining < 10 else "llm_response_failed"
            return
        state["turns"] += 1
        turn: dict[str, Any] = {"response": response, "calls": []}
        state["trajectory"].append(turn)

        calls = parse_tool_calls(response)
        errors = {"parse_errors": 0, "tool_errors": 0, "tool_timeouts": 0}
        observations: list[str] = []
        done = False
        if not calls:
            errors["parse_errors"] += 1
            observations.append(NO_CALL_OBSERVATION)
        if len(calls) > config["max_calls_per_turn"]:
            observations.append(
                f"Only the first {config['max_calls_per_turn']} of {len(calls)} function calls were executed."
            )
            calls = calls[: config["max_calls_per_turn"]]

        for index, call in enumerate(calls, start=1):
            name, args = call["function"], call["arguments"]
            missing = [param for param in _REQUIRED.get(name, []) if param not in args]
            if name not in tool_names:
                errors["parse_errors"] += 1
                observation = f"Error: unknown function {name!r}. Available: {', '.join(sorted(tool_names))}."
            elif missing:
                errors["parse_errors"] += 1
                observation = f"Error: {name} is missing required parameter(s): {', '.join(missing)}."
            elif name == "return_answer":
                state["final_answer"] = args["answer"].strip()
                state["terminate_reason"] = "return_answer"
                done = True
                turn["calls"].append({"function": name, "arguments": args})
                break
            else:
                state["tool_calls"] += 1
                observation, outcome = await _execute(trial, name, args, state, config)
                if outcome:
                    errors[outcome] += 1
            turn["calls"].append({"function": name, "arguments": args, "observation": observation})
            observations.append(f"[{index}] {name}:\n{observation}" if len(calls) > 1 else observation)

        for key, count in errors.items():
            state[key] += count
        apply_action_process_penalty(agent, state, turn, config, **errors, enabled=shaping)
        if done:
            return
        agent.append(
            {"role": "user", "content": _truncate("\n\n".join(observations), config["max_turn_observation_chars"])}
        )
    state["terminate_reason"] = "max_turns"


async def _execute(trial, name: str, args: dict, state: dict, config: dict) -> tuple[str, str | None]:
    """Run one tool call. Returns (observation, error counter name or None)."""
    budget = config["tool_total_budget"] - state["tool_seconds"]
    if budget <= 0:
        return "Error: the tool time budget for this task is exhausted. Call return_answer.", "tool_timeouts"
    timeout = max(1.0, min(config["tool_timeout"], budget))
    started = time.perf_counter()
    try:
        result = await trial.execute(name, args, timeout=timeout)
    except (SandboxError, asyncio.TimeoutError) as exc:
        # Other transport errors are infrastructure failures and end the episode.
        message = str(exc) or type(exc).__name__
        outcome = (
            "tool_timeouts" if "TimeoutError" in message or isinstance(exc, asyncio.TimeoutError) else "tool_errors"
        )
        return f"Error: {_truncate(message, config['max_observation_chars'])}", outcome
    finally:
        state["tool_seconds"] += time.perf_counter() - started
    if not isinstance(result, str):
        result = json.dumps(result, ensure_ascii=False, default=str)
    return _truncate(result, config["max_observation_chars"]) or "(no output)", None


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head]}\n... [{len(text) - limit} characters truncated] ...\n{text[-tail:]}"


async def _compute_reward(data: dict, extra_info: dict, trial, answer: str) -> tuple[float, bool, str]:
    ground_truth = (data.get("reward_model") or {}).get("ground_truth")
    mode = extra_info.get("reward_mode") or ("match" if ground_truth is not None else "sandbox")
    if mode == "match":
        score = await asyncio.to_thread(match_score, answer, str(ground_truth))
        return score, score >= 1.0, ""
    if trial is None:
        return 0.0, False, "no sandbox trial"
    try:
        result = await trial.reward(answer=answer)
    except Exception as exc:
        return 0.0, False, f"reward_error: {type(exc).__name__}: {exc}"
    return float(result["reward"]), bool(result["passed"]), str(result.get("details", {}).get("reason", ""))


def match_score(answer: str, ground_truth: str) -> float:
    """Exact match after light normalization, then Math-Verify equivalence."""

    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower().replace(",", "")).rstrip(".")

    if not answer:
        return 0.0
    if normalize(answer) == normalize(ground_truth):
        return 1.0
    from verl.utils.reward_score.math_verify import compute_score

    return float(compute_score(answer, ground_truth))


async def _heartbeat(trial, interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            await trial.heartbeat()
        except Exception:
            pass


async def _cleanup(trial, session) -> None:
    for closer in (trial and trial.destroy, session and session.close):
        if closer is None:
            continue
        try:
            await closer()
        except Exception:
            pass


def _write_trace(path: str, state: dict, metrics: dict, reward_reason: str, messages: list) -> None:
    """Append one JSON line per rollout. Never affects training."""
    if not path:
        return
    try:
        record = {
            "time": time.time(),
            **{key: state[key] for key in ("task_id", "final_answer", "terminate_reason", "error", "trajectory")},
            "reward_reason": reward_reason,
            "metrics": metrics,
            "messages": messages,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode("utf-8")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            view = memoryview(payload)
            while view:
                view = view[os.write(fd, view) :]
        finally:
            os.close(fd)
    except Exception:
        return
