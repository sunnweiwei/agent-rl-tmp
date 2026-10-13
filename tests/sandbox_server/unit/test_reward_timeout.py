from types import SimpleNamespace

import pytest

from sandbox_server.schemas import RewardRequest
from sandbox_server.server import compute_reward


class _CapturingRewardManager:
    def __init__(self):
        self.calls = []

    async def compute_reward(self, trial_id, answer=None, timeout=None):
        self.calls.append((trial_id, answer, timeout))
        return SimpleNamespace(reward=0.25, passed=False, details={"timeout": timeout})


@pytest.mark.asyncio
async def test_reward_endpoint_passes_timeout_to_manager():
    manager = _CapturingRewardManager()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(sandbox_manager=manager)))

    result = await compute_reward("trial1", RewardRequest(answer="done", timeout=12.5), request)

    assert result.reward == 0.25
    assert result.passed is False
    assert result.details["timeout"] == 12.5
    assert manager.calls == [("trial1", "done", 12.5)]
