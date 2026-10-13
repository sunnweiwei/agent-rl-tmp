import pytest
from pydantic import ValidationError

from sandbox_server.schemas import (
    ActionRequest,
    ActionResponse,
    CapacityView,
    HealthView,
    RewardRequest,
    RewardView,
    SessionCreate,
    TrialCreate,
    TrialState,
    TrialView,
)


def test_trial_state_values():
    assert TrialState.READY.value == "ready"
    assert TrialState.ACTIVE.value == "active"
    assert TrialState.DESTROYED.value == "destroyed"


def test_session_create_minimal():
    s = SessionCreate(profile="nsjail")
    assert s.profile == "nsjail"
    assert s.metadata == {}


def test_trial_create_defaults():
    t = TrialCreate()
    assert t.task_id == ""
    assert t.metadata == {}
    assert t.lease_seconds is None


def test_action_request_rejects_zero_timeout():
    with pytest.raises(ValidationError):
        ActionRequest(tool="bash", timeout=0)


def test_action_request_rejects_negative_timeout():
    with pytest.raises(ValidationError):
        ActionRequest(tool="bash", timeout=-1)


def test_action_request_valid():
    a = ActionRequest(tool="bash", args={"command": "ls"}, timeout=30)
    assert a.tool == "bash"
    assert a.timeout == 30


def test_action_response_success():
    r = ActionResponse(success=True, result={"data": [1, 2]})
    assert r.success is True
    assert r.error is None


def test_action_response_error():
    r = ActionResponse(success=False, error="RuntimeError: oops")
    assert r.success is False
    assert r.result is None


def test_reward_request_none_answer():
    r = RewardRequest()
    assert r.answer is None
    assert r.timeout is None


def test_reward_request_with_answer():
    r = RewardRequest(answer="42")
    assert r.answer == "42"


def test_reward_request_rejects_non_positive_timeout():
    with pytest.raises(ValidationError):
        RewardRequest(timeout=0)
    with pytest.raises(ValidationError):
        RewardRequest(timeout=-1)


def test_reward_request_custom_timeout():
    r = RewardRequest(timeout=123)
    assert r.timeout == 123


def test_reward_view():
    v = RewardView(reward=1.0, passed=True, details={"score": 1.0})
    assert v.reward == 1.0
    assert v.passed is True


def test_health_view():
    h = HealthView(status="ok", backends=["nsjail", "docker"], profiles=["nsjail"])
    assert "nsjail" in h.backends


def test_capacity_view():
    c = CapacityView(
        profile="nsjail",
        live_trials=10,
        max_live_trials=1024,
        running_actions=5,
        max_running_actions=128,
        queued_actions=2,
    )
    assert c.live_trials == 10


def test_trial_state_json_roundtrip():
    v = TrialView(
        trial_id="abc",
        session_id="sess",
        task_id="t1",
        state=TrialState.READY,
        lease_deadline=100.0,
    )
    data = v.model_dump()
    assert data["state"] == "ready"
    restored = TrialView(**data)
    assert restored.state == TrialState.READY
