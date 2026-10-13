# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Math reasoning agent functions."""

from __future__ import annotations

from typing import Any

from agents.core.agent import Agent, TaskContext
from verl.utils.reward_score import default_compute_score


async def agent_loop(data: dict[str, Any], context: TaskContext):
    raw_prompt = data.get("raw_prompt") or data.get("prompt") or []
    chat = [dict(turn) for turn in list(raw_prompt)]
    agent = Agent(context, chat, prompt_turn=len(chat))
    response = await agent.step()
    response = response or ""

    reward_model = data.get("reward_model") or {}
    extra_info = dict(data.get("extra_info") or {})
    ground_truth = reward_model.get("ground_truth")
    if ground_truth is None:
        ground_truth = extra_info.get("answer")
    data_source = str(data.get("data_source", "openai/gsm8k"))

    score_result = default_compute_score(
        data_source=data_source,
        solution_str=response,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    if isinstance(score_result, dict):
        reward = float(score_result["score"])
        reward_extra_info = dict(score_result)
    else:
        reward = float(score_result)
        reward_extra_info = {"score": reward}
    reward_extra_info.update(
        {
            "all/score": reward,
            "math/acc": reward,
            "math/response_tokens": int(sum(agent.response_mask)),
        }
    )
    return await agent.get_agent_output(reward, extra_info=reward_extra_info)
