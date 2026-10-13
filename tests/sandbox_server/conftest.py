import pytest

from sandbox_server.isolation.base import ExecutionResult, IsolationBackend, TrialHandle


class MockBackend(IsolationBackend):
    """Records all calls for assertion. Returns configurable results."""

    name = "mock"

    def __init__(self, execute_result=None):
        self.created: list[str] = []
        self.executed: list[tuple[str, list[str]]] = []
        self.destroyed: list[str] = []
        self._execute_result = execute_result or ExecutionResult(0, "", "")

    @property
    def available(self) -> bool:
        return True

    async def create_trial(self, trial_id, profile, workspace, task_config=None):
        self.created.append(trial_id)
        return TrialHandle(trial_id=trial_id, workspace=workspace)

    async def execute(self, handle, profile, argv, timeout, env=None):
        self.executed.append((handle.trial_id, argv))
        return self._execute_result

    async def destroy(self, handle):
        self.destroyed.append(handle.trial_id)


class UnavailableBackend(MockBackend):
    @property
    def available(self) -> bool:
        return False


class FailingCreateBackend(MockBackend):
    async def create_trial(self, trial_id, profile, workspace, task_config=None):
        raise RuntimeError("backend create failed")


@pytest.fixture
def workspace_root(tmp_path):
    return tmp_path / "workspaces"


@pytest.fixture
def mock_backend():
    return MockBackend()


@pytest.fixture
def unavailable_backend():
    return UnavailableBackend()


@pytest.fixture
def failing_create_backend():
    return FailingCreateBackend()
