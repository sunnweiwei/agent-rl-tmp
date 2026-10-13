# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Thin verl adapter that routes each sample to a function-style agent loop.

An agent loop is ``async def agent_loop(data: dict, context: TaskContext) -> AgentLoopOutput``.
Samples are routed by their ``data_source``; register new agents in ``AGENT_LOOPS``.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

from agents.core.agent import CallLLM, TaskContext
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# data_source -> "module:function"
AGENT_LOOPS = {
    "sandbox_agent": "agents.sandbox_agent.agent:agent_loop",
    "math": "agents.math.agent:agent_loop",
}


def get_agent_loop(data_source: str, default: str = "math"):
    target = AGENT_LOOPS.get(data_source) or AGENT_LOOPS[default]
    module_name, function_name = target.split(":")
    return getattr(importlib.import_module(module_name), function_name)


@register("agent_hub")
class AgentHubLoop(AgentLoopBase):
    """Route each sample to a task-specific async agent function."""

    def __init__(self, *args, **kwargs):
        self.agent_options = dict(kwargs)
        self.agent_options.pop("tools", None)
        super().__init__(*args, **kwargs)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        llm_client = CallLLM(
            server_manager=self.server_manager,
            tokenizer=self.tokenizer,
            rollout_config=self.rollout_config,
            loop=self.loop,
            sampling_params=sampling_params,
        )
        context = TaskContext(
            config=self.config,
            global_step=int(kwargs.get("global_step", 0) or 0),
            is_train=bool(kwargs.get("is_train", True)),
            tokenizer=self.tokenizer,
            agent_loop=self,
            llm_client=llm_client,
            sampling_params=sampling_params,
            agent_options=self.agent_options,
        )
        agent_loop_fn = get_agent_loop(
            str(kwargs.get("data_source", "")), self.agent_options.get("default_agent", "math")
        )
        return await agent_loop_fn(dict(kwargs), context)
