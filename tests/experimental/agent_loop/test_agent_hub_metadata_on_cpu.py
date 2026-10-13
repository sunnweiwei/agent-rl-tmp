import aiohttp
import pytest

from agents.core.agent import Agent, _call_with_transport_retries


def test_agent_accumulates_weight_version_range_and_lora_status():
    agent = object.__new__(Agent)
    agent.extra_fields = {}

    agent._accumulate_completion_extra_fields(
        {"global_steps": 3, "lora_applied": True, "prefix_cache_query_tokens": 10, "num_cached_tokens": 4}
    )
    agent._accumulate_completion_extra_fields(
        {"global_steps": 4, "lora_applied": False, "prefix_cache_query_tokens": 15, "num_cached_tokens": 7}
    )

    assert agent.extra_fields["min_global_steps"] == 3
    assert agent.extra_fields["max_global_steps"] == 4
    assert agent.extra_fields["global_steps"] == 4
    assert agent.extra_fields["lora_applied"] is False
    assert agent.extra_fields["prefix_cache_query_tokens"] == 25
    assert agent.extra_fields["prefix_cache_cached_tokens"] == 11
    assert agent.extra_fields["prefix_cache_queries"] == 2


@pytest.mark.asyncio
async def test_rollout_transport_connection_errors_are_retried(monkeypatch):
    monkeypatch.setenv("VERL_ROLLOUT_TRANSPORT_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("VERL_ROLLOUT_TRANSPORT_RETRY_BASE_SECONDS", "0")
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise aiohttp.ServerDisconnectedError("transient disconnect")
        return "ok"

    assert await _call_with_transport_retries(operation) == "ok"
    assert calls == 3


@pytest.mark.asyncio
async def test_rollout_transport_non_connection_errors_are_not_retried(monkeypatch):
    monkeypatch.setenv("VERL_ROLLOUT_TRANSPORT_MAX_ATTEMPTS", "4")
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        raise ValueError("bad request")

    with pytest.raises(ValueError, match="bad request"):
        await _call_with_transport_retries(operation)
    assert calls == 1
