from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_REWARD_TIMEOUT_SECONDS = 600.0
DEFAULT_REWARD_HTTP_TIMEOUT_SECONDS = 1210.0
REWARD_HTTP_TIMEOUT_BUFFER_SECONDS = 10.0


class TrialState(str, Enum):
    ALLOCATING = "allocating"
    READY = "ready"
    ACTIVE = "active"
    EVALUATING = "evaluating"
    CLEANING = "cleaning"
    DESTROYED = "destroyed"
    FAILED = "failed"


class SessionCreate(BaseModel):
    profile: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionView(BaseModel):
    session_id: str
    profile: str
    profile_version: str
    created_at: float


class TrialCreate(BaseModel):
    # Relative task directory under the daemon's task root. Empty means a
    # scratch trial with an empty workspace and no server-side reward.
    task_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    lease_seconds: float | None = None


class TrialView(BaseModel):
    trial_id: str
    session_id: str
    task_id: str
    state: TrialState
    lease_deadline: float


class ActionRequest(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    timeout: float = Field(default=120.0, gt=0)


class ActionResponse(BaseModel):
    success: bool
    result: Any | None = None
    error: str | None = None
    queue_wait_seconds: float = 0.0
    execution_seconds: float = 0.0


class RewardRequest(BaseModel):
    answer: str | None = None
    # None means the server uses the task/profile default.
    timeout: float | None = Field(default=None, gt=0)


class RewardView(BaseModel):
    reward: float
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


class CapacityView(BaseModel):
    profile: str
    live_trials: int
    max_live_trials: int
    running_actions: int
    max_running_actions: int
    queued_actions: int
    max_queued_actions: int | None = None
    running_actions_by_tool: dict[str, int] = Field(default_factory=dict)
    queued_actions_by_tool: dict[str, int] = Field(default_factory=dict)
    max_running_actions_by_tool: dict[str, int] = Field(default_factory=dict)


class HealthView(BaseModel):
    status: str
    backends: list[str]
    profiles: list[str]
