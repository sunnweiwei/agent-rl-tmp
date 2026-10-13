"""Turn-local process rewards (action error penalties) and context-limit penalty."""

from __future__ import annotations

from typing import Any


def action_error_penalty(
    *,
    parse_errors: int,
    tool_errors: int,
    tool_timeouts: int,
    parse_error_penalty: float,
    tool_error_penalty: float,
    tool_timeout_penalty: float,
    max_penalty: float,
) -> float:
    """Return a capped, presence-based penalty for one assistant turn.

    Multiple tool calls of the same error type in one turn share one penalty so
    the shaping signal does not reward emitting fewer calls per turn.
    """
    magnitude = (
        int(int(parse_errors) > 0) * max(0.0, float(parse_error_penalty))
        + int(int(tool_errors) > 0) * max(0.0, float(tool_error_penalty))
        + int(int(tool_timeouts) > 0) * max(0.0, float(tool_timeout_penalty))
    )
    cap = max(0.0, float(max_penalty))
    if cap > 0.0:
        magnitude = min(magnitude, cap)
    return -magnitude


def context_limit_penalty(
    remaining_tokens: int | None,
    *,
    buffer_tokens: int,
    max_penalty: float,
    context_full: bool = False,
) -> float:
    """Linearly ramp from zero to ``-max_penalty`` near the context limit."""
    maximum = max(0.0, float(max_penalty))
    if maximum == 0.0:
        return 0.0
    if context_full:
        return -maximum
    if remaining_tokens is None:
        return 0.0

    buffer = max(0, int(buffer_tokens))
    if buffer == 0:
        return -maximum if int(remaining_tokens) <= 0 else 0.0

    remaining = max(0, int(remaining_tokens))
    if remaining >= buffer:
        return 0.0
    return -maximum * (buffer - remaining) / buffer


def apply_action_process_penalty(
    rollout_agent: Any,
    state: dict[str, Any],
    turn: dict[str, Any],
    config: dict[str, Any],
    *,
    parse_errors: int,
    tool_errors: int,
    tool_timeouts: int,
    enabled: bool,
) -> float:
    """Attach one turn-local penalty and record its diagnostics."""
    if not enabled or not bool(config.get("process_rewards_enabled", False)):
        return 0.0

    penalty = action_error_penalty(
        parse_errors=parse_errors,
        tool_errors=tool_errors,
        tool_timeouts=tool_timeouts,
        parse_error_penalty=float(config.get("parse_error_penalty", 0.0)),
        tool_error_penalty=float(config.get("tool_error_penalty", 0.0)),
        tool_timeout_penalty=float(config.get("tool_timeout_penalty", 0.0)),
        max_penalty=float(config.get("action_error_penalty_cap", 0.0)),
    )
    if penalty == 0.0:
        return 0.0

    marked_tokens = int(rollout_agent.add_process_reward_to_last_action(penalty))
    if marked_tokens <= 0:
        return 0.0

    state["process_reward_sum"] = float(state.get("process_reward_sum", 0.0)) + penalty
    state["process_reward_actions"] = int(state.get("process_reward_actions", 0)) + 1
    turn["process_reward"] = round(penalty, 6)
    turn["process_reward_tokens"] = marked_tokens
    turn["process_reward_errors"] = {
        "parse": max(0, int(parse_errors)),
        "tool": max(0, int(tool_errors)),
        "timeout": max(0, int(tool_timeouts)),
    }
    return penalty
