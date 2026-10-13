# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Shared multi-turn agent utilities for project-local agent loops."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar

import aiohttp
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.utils.chat_template import apply_chat_template
from verl.utils.profiler import simple_timer
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _rollout_transport_retry_config() -> tuple[int, float]:
    try:
        max_attempts = max(1, int(os.getenv("VERL_ROLLOUT_TRANSPORT_MAX_ATTEMPTS", "4")))
    except ValueError:
        max_attempts = 4
    try:
        base_delay_seconds = max(0.0, float(os.getenv("VERL_ROLLOUT_TRANSPORT_RETRY_BASE_SECONDS", "0.5")))
    except ValueError:
        base_delay_seconds = 0.5
    return max_attempts, base_delay_seconds


async def _call_with_transport_retries(operation: Callable[[], Awaitable[_T]]) -> _T:
    """Retry connection failures that happen before a rollout response is received."""
    max_attempts, base_delay_seconds = _rollout_transport_retry_config()
    for attempt in range(1, max_attempts + 1):
        try:
            return await operation()
        except aiohttp.ClientConnectionError as exc:
            if attempt == max_attempts:
                raise
            delay_seconds = base_delay_seconds * (2 ** (attempt - 1))
            logger.warning(
                "Rollout transport error on attempt %d/%d (%s: %s); retrying in %.2fs",
                attempt,
                max_attempts,
                type(exc).__name__,
                exc,
                delay_seconds,
            )
            await asyncio.sleep(delay_seconds)

    raise AssertionError("unreachable")


@dataclass
class Completion:
    content: str
    token_ids: list[int]
    log_probs: list[float]
    routed_experts: Any = None
    metrics: dict[str, float] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict)


class CallLLM:
    """Verl rollout-server client used by function-style agents."""

    def __init__(
        self,
        *,
        server_manager: LLMServerClient,
        tokenizer: PreTrainedTokenizer,
        rollout_config: Any,
        loop: Any,
        sampling_params: dict[str, Any],
    ):
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.rollout_config = rollout_config
        self.loop = loop
        self.sampling_params = dict(sampling_params)

    async def create_completion(
        self,
        prompt_ids: list[int],
        *,
        uid: str | None = None,
        sampling_params: dict[str, Any] | None = None,
        max_new_tokens: int | None = None,
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
        audio_data: list[Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> Completion | None:
        params = dict(self.sampling_params)
        if sampling_params:
            params.update(sampling_params)
        if max_new_tokens is not None:
            params["max_tokens"] = max_new_tokens

        metrics: dict[str, float] = {}
        with simple_timer("generate_sequences", metrics):

            async def generate() -> TokenOutput:
                return await self.server_manager.generate(
                    request_id=uid or uuid.uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=params,
                    image_data=image_data,
                    video_data=video_data,
                    audio_data=audio_data,
                    mm_processor_kwargs=mm_processor_kwargs,
                )

            output = await _call_with_transport_retries(generate)
        if output is None or len(output.token_ids) == 0:
            return None
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        token_ids = list(output.token_ids)
        text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(token_ids, skip_special_tokens=True),
        )
        raw_log_probs = output.log_probs
        if raw_log_probs is None or len(raw_log_probs) == 0:
            log_probs = [0.0] * len(token_ids)
        else:
            log_probs = list(raw_log_probs)
            if len(log_probs) < len(token_ids):
                log_probs.extend([0.0] * (len(token_ids) - len(log_probs)))
            elif len(log_probs) > len(token_ids):
                log_probs = log_probs[: len(token_ids)]

        extra_fields = dict(output.extra_fields or {})
        extra_fields["prefix_cache_query_tokens"] = len(prompt_ids)

        return Completion(
            content=text,
            token_ids=token_ids,
            log_probs=log_probs,
            routed_experts=output.routed_experts,
            metrics=metrics,
            extra_fields=extra_fields,
        )


@dataclass
class TaskContext:
    config: DictConfig
    global_step: int
    is_train: bool
    tokenizer: PreTrainedTokenizer
    agent_loop: Any
    llm_client: CallLLM
    sampling_params: dict[str, Any]
    agent_options: dict[str, Any] = field(default_factory=dict)

    @property
    def rollout_config(self) -> Any:
        return self.config.actor_rollout_ref.rollout

    @property
    def prompt_length(self) -> int:
        return int(self.rollout_config.prompt_length)

    @property
    def response_length(self) -> int:
        return int(self.rollout_config.response_length)

    @property
    def apply_chat_template_kwargs(self) -> dict[str, Any]:
        return dict(getattr(self.agent_loop, "apply_chat_template_kwargs", {}) or {})

    async def tokenize_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        truncate_to_prompt_length: bool = False,
    ) -> list[int]:
        tokenized = await self.agent_loop.loop.run_in_executor(
            None,
            lambda: apply_chat_template(
                self.tokenizer,
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )
        token_ids = normalize_token_ids(tokenized)
        if truncate_to_prompt_length and len(token_ids) > self.prompt_length:
            token_ids = token_ids[-self.prompt_length :]
        return token_ids

    def tokenize_chat_sync(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        truncate_to_prompt_length: bool = False,
    ) -> list[int]:
        tokenized = apply_chat_template(
            self.tokenizer,
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )
        token_ids = normalize_token_ids(tokenized)
        if truncate_to_prompt_length and len(token_ids) > self.prompt_length:
            token_ids = token_ids[-self.prompt_length :]
        return token_ids


class _ImmediateAwaitable:
    def __init__(self, value=None):
        self.value = value

    def __await__(self):
        if False:
            yield
        return self.value


class Agent:
    """A multi-turn trajectory recorder.

    A one-turn math rollout is just an Agent that calls ``step`` once.  Tool and
    runtime-backed agents can interleave ``step`` with ``append`` observations.
    """

    def __init__(
        self,
        context: TaskContext,
        chat: list[dict[str, Any]],
        *,
        prompt_turn: int | None = None,
        append_eos_if_missing: bool = False,
    ):
        self.context = context
        self.chat = copy.deepcopy(chat)
        self.prompt_turn = len(self.chat) if prompt_turn is None else int(prompt_turn)
        self.append_eos_if_missing = append_eos_if_missing
        self.context_uid = uuid.uuid4().hex

        self.response_ids: list[int] = []
        self.response_logprobs: list[float] = []
        self.response_mask: list[int] = []
        self.process_advantages: list[float] = []
        self.metrics = AgentLoopMetrics()
        self.extra_fields: dict[str, Any] = {}
        self.routed_experts = None
        self._prompt_ids: list[int] | None = None
        self._prompt_reserve_tokens = -1
        self._context_ids: list[int] | None = None
        self._last_action_span: tuple[int, int] | None = None

    async def prompt_ids(self, *, reserve_tokens: int = 0) -> list[int]:
        reserve_tokens = int(reserve_tokens)
        if self._prompt_ids is None or reserve_tokens > self._prompt_reserve_tokens:
            prompt_ids = await self.context.tokenize_chat(
                self.chat[: self.prompt_turn],
                add_generation_prompt=False,
                truncate_to_prompt_length=False,
            )
            max_prompt_tokens = max(0, self.context.prompt_length - int(reserve_tokens))
            if max_prompt_tokens <= 0:
                prompt_ids = []
            elif len(prompt_ids) > max_prompt_tokens:
                prompt_ids = prompt_ids[-max_prompt_tokens:]
            self._prompt_ids = prompt_ids
            self._prompt_reserve_tokens = reserve_tokens
        return self._prompt_ids

    async def context_ids(self, *, reserve_tokens: int = 0) -> list[int]:
        """Return the authoritative current context without a new generation prompt."""
        if self._context_ids is None:
            self._context_ids = list(await self.prompt_ids(reserve_tokens=reserve_tokens))
        return list(self._context_ids)

    async def generation_prompt_ids(self) -> list[int]:
        without_generation_prompt = await self.context.tokenize_chat(
            self.chat,
            add_generation_prompt=False,
            truncate_to_prompt_length=False,
        )
        with_generation_prompt = await self.context.tokenize_chat(
            self.chat,
            add_generation_prompt=True,
            truncate_to_prompt_length=False,
        )
        if with_generation_prompt[: len(without_generation_prompt)] == without_generation_prompt:
            return with_generation_prompt[len(without_generation_prompt) :]
        return []

    async def step(
        self, *, max_new_tokens: int | None = None, sampling_params: dict[str, Any] | None = None
    ) -> str | None:
        generation_prompt_ids = await self.generation_prompt_ids()
        context_ids = await self.context_ids(reserve_tokens=len(generation_prompt_ids))
        remaining_response_tokens = self.context.response_length - len(self.response_ids) - len(generation_prompt_ids)
        if remaining_response_tokens <= 0:
            return None
        request_max_new_tokens = remaining_response_tokens
        if max_new_tokens is not None:
            request_max_new_tokens = min(int(max_new_tokens), remaining_response_tokens)

        prompt_ids = context_ids + generation_prompt_ids
        completion = await self.context.llm_client.create_completion(
            prompt_ids,
            uid=self.context_uid,
            sampling_params=sampling_params,
            max_new_tokens=request_max_new_tokens,
        )
        if completion is None:
            return None

        response_token_ids = list(completion.token_ids)
        response_logprobs = list(completion.log_probs)
        if self.append_eos_if_missing:
            eos_token_id = self.context.tokenizer.eos_token_id
            if eos_token_id is not None and (not response_token_ids or response_token_ids[-1] != eos_token_id):
                response_token_ids.append(int(eos_token_id))
                response_logprobs.append(0.0)

        action_start = len(self.response_ids) + len(generation_prompt_ids)
        action_end = action_start + len(completion.token_ids)
        self.response_ids.extend(generation_prompt_ids + response_token_ids)
        self.response_logprobs.extend([0.0] * len(generation_prompt_ids) + response_logprobs)
        self.response_mask.extend([0] * len(generation_prompt_ids) + [1] * len(completion.token_ids))
        if len(response_token_ids) > len(completion.token_ids):
            self.response_mask.append(0)
        self._context_ids = context_ids + generation_prompt_ids + response_token_ids
        self._last_action_span = (action_start, action_end)
        self._normalize_response_alignment()

        self.metrics.generate_sequences += completion.metrics.get("generate_sequences", 0.0)
        self.metrics.num_preempted = int(completion.metrics.get("num_preempted", self.metrics.num_preempted))
        self._accumulate_completion_extra_fields(completion.extra_fields)
        if completion.routed_experts is not None:
            self.routed_experts = completion.routed_experts
        self.chat.append({"role": "assistant", "content": completion.content})
        return completion.content

    def append(self, turn: dict[str, Any], *, trainable: bool = False) -> _ImmediateAwaitable:
        before_chat = copy.deepcopy(self.chat)
        before_ids = self.context.tokenize_chat_sync(
            before_chat,
            add_generation_prompt=False,
            truncate_to_prompt_length=False,
        )
        self.chat.append(copy.deepcopy(turn))
        after_ids = self.context.tokenize_chat_sync(
            self.chat,
            add_generation_prompt=False,
            truncate_to_prompt_length=False,
        )
        if after_ids[: len(before_ids)] == before_ids:
            delta = after_ids[len(before_ids) :]
        else:
            context_ids = list(self._context_ids) if self._context_ids is not None else list(before_ids)
            if after_ids[: len(context_ids)] == context_ids:
                delta = after_ids[len(context_ids) :]
            else:
                delta = self.context.tokenize_chat_sync(
                    [turn],
                    add_generation_prompt=False,
                    truncate_to_prompt_length=False,
                )
        self.response_ids.extend(delta)
        self.response_logprobs.extend([0.0] * len(delta))
        self.response_mask.extend([1 if trainable else 0] * len(delta))
        context_ids = list(self._context_ids) if self._context_ids is not None else list(before_ids)
        self._context_ids = context_ids + delta
        self._normalize_response_alignment()
        return _ImmediateAwaitable()

    def add_process_reward_to_last_action(self, reward: float) -> int:
        """Apply an action-local advantage to the latest generated assistant tokens."""
        if self._last_action_span is None or float(reward) == 0.0:
            return 0
        self._normalize_response_alignment()
        start, end = self._last_action_span
        end = min(end, len(self.process_advantages))
        marked = 0
        for index in range(max(0, start), end):
            if self.response_mask[index]:
                self.process_advantages[index] += float(reward)
                marked += 1
        return marked

    async def get_agent_output(
        self,
        reward_score: float | None,
        *,
        extra_info: dict[str, Any] | None = None,
        turn_scores: list[Any] | None = None,
        tool_rewards: list[Any] | None = None,
    ) -> AgentLoopOutput:
        prompt_ids = await self.prompt_ids()
        response_length = self.context.response_length
        self._normalize_response_alignment()
        extra_fields = {
            "turn_scores": turn_scores or [],
            "tool_rewards": tool_rewards or [],
            "reward_extra_info": extra_info or {},
            "messages": self.chat,
        }
        extra_fields.update(self.extra_fields)
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=self.response_ids[:response_length],
            response_mask=self.response_mask[:response_length],
            process_advantages=self.process_advantages[:response_length],
            response_logprobs=self.response_logprobs[:response_length] if self.response_logprobs else None,
            routed_experts=self.routed_experts,
            multi_modal_data={},
            reward_score=reward_score,
            num_turns=len(self.chat),
            metrics=self.metrics,
            extra_fields=extra_fields,
        )

    def _accumulate_completion_extra_fields(self, extra_fields: dict[str, Any]) -> None:
        if not extra_fields:
            return

        min_global_steps = extra_fields.get("min_global_steps", extra_fields.get("global_steps"))
        max_global_steps = extra_fields.get("max_global_steps", extra_fields.get("global_steps"))
        if min_global_steps is not None:
            current_min = self.extra_fields.get("min_global_steps", min_global_steps)
            self.extra_fields["min_global_steps"] = min(int(current_min), int(min_global_steps))
        if max_global_steps is not None:
            current_max = self.extra_fields.get("max_global_steps", max_global_steps)
            self.extra_fields["max_global_steps"] = max(int(current_max), int(max_global_steps))
            self.extra_fields["global_steps"] = self.extra_fields["max_global_steps"]
        if "lora_applied" in extra_fields:
            self.extra_fields["lora_applied"] = bool(self.extra_fields.get("lora_applied", True)) and bool(
                extra_fields["lora_applied"]
            )

        query_tokens = extra_fields.get("prefix_cache_query_tokens")
        cached_tokens = extra_fields.get("num_cached_tokens")
        if query_tokens is not None:
            self.extra_fields["prefix_cache_query_tokens"] = int(
                self.extra_fields.get("prefix_cache_query_tokens", 0)
            ) + int(query_tokens)
        if cached_tokens is not None:
            self.extra_fields["prefix_cache_cached_tokens"] = int(
                self.extra_fields.get("prefix_cache_cached_tokens", 0)
            ) + int(cached_tokens)
            self.extra_fields["prefix_cache_queries"] = int(self.extra_fields.get("prefix_cache_queries", 0)) + 1

        for key in ("spec_num_draft_tokens", "spec_num_accepted_tokens", "spec_num_verify_steps"):
            if key in extra_fields and extra_fields[key] is not None:
                self.extra_fields[key] = int(self.extra_fields.get(key, 0)) + int(extra_fields[key])

    def _normalize_response_alignment(self) -> None:
        target = len(self.response_ids)
        if len(self.response_mask) < target:
            self.response_mask.extend([0] * (target - len(self.response_mask)))
        elif len(self.response_mask) > target:
            self.response_mask = self.response_mask[:target]
        if len(self.response_logprobs) < target:
            self.response_logprobs.extend([0.0] * (target - len(self.response_logprobs)))
        elif len(self.response_logprobs) > target:
            self.response_logprobs = self.response_logprobs[:target]
        if len(self.process_advantages) < target:
            self.process_advantages.extend([0.0] * (target - len(self.process_advantages)))
        elif len(self.process_advantages) > target:
            self.process_advantages = self.process_advantages[:target]

    def think_format_correct(self) -> int:
        enable_think = os.getenv("TURNOFF_THINK", "1").lower() in ("0", "false", "no")
        for turn in self.chat[self.prompt_turn :]:
            if turn.get("role") != "assistant":
                continue
            content = (turn.get("content") or "").strip()
            starts_think = content.startswith("<think>") or content.startswith("<seed:think>")
            if enable_think:
                if not starts_think:
                    return 0
                if "</think>" not in content and "</seed:think>" not in content:
                    return 0
            elif starts_think:
                return 0
        return 1

    async def react(
        self,
        run_action,
        *,
        max_turns: int = 64,
        session_timeout: float = 3600.0,
        should_continue=None,
    ) -> dict[str, Any]:
        should_continue = should_continue or (lambda response: True)
        start_time = time.time()
        last_response = None
        iterations = 0
        while iterations < max_turns and time.time() - start_time <= session_timeout:
            iterations += 1
            response = await self.step()
            if response is None:
                break
            last_response = response
            if not should_continue(response):
                break
            observation = await run_action(response)
            if observation is None:
                break
            await self.append({"role": "user", "content": observation})
        return {"last_response": last_response, "iterations": iterations}
