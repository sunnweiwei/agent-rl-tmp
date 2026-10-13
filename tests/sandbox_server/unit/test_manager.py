import asyncio

import pytest
import pytest_asyncio

from sandbox_server.isolation.base import ExecutionResult
from sandbox_server.manager import SandboxManager
from sandbox_server.profiles import SandboxProfile
from sandbox_server.schemas import TrialState
from tests.sandbox_server.conftest import (
    FailingCreateBackend,
    MockBackend,
    UnavailableBackend,
)


@pytest.fixture
def profile():
    return SandboxProfile(
        name="test",
        backend_type="mock",
        enabled_tools=frozenset({"bash", "return_answer"}),
        max_live_trials=5,
        max_running_actions=2,
        lease_seconds=60.0,
    )


@pytest_asyncio.fixture
async def started_manager(workspace_root, mock_backend, profile):
    profiles = {"test": profile}
    backends = {"mock": mock_backend}
    mgr = SandboxManager(profiles, backends, workspace_root, lease_check_interval=100)
    await mgr.start()
    yield mgr
    await mgr.close()


# --- Session lifecycle ---


@pytest.mark.asyncio
async def test_create_session(started_manager):
    session = await started_manager.create_session("test", {"user": "alice"})
    assert session.session_id
    assert session.profile.name == "test"
    assert session.metadata == {"user": "alice"}


@pytest.mark.asyncio
async def test_create_session_unknown_profile(started_manager):
    with pytest.raises(KeyError, match="unknown profile"):
        await started_manager.create_session("nonexistent", {})


@pytest.mark.asyncio
async def test_create_session_unavailable_backend(workspace_root):
    profile = SandboxProfile(name="test", backend_type="mock")
    mgr = SandboxManager(
        {"test": profile},
        {"mock": UnavailableBackend()},
        workspace_root,
    )
    await mgr.start()
    with pytest.raises(RuntimeError, match="not available"):
        await mgr.create_session("test", {})
    await mgr.close()


@pytest.mark.asyncio
async def test_destroy_session_cascades(started_manager, mock_backend):
    session = await started_manager.create_session("test", {})
    await started_manager.create_trial(session.session_id, "", {})
    await started_manager.create_trial(session.session_id, "", {})
    assert len(started_manager.trials) == 2

    await started_manager.destroy_session(session.session_id)
    assert len(started_manager.trials) == 0
    assert len(started_manager.sessions) == 0
    assert len(mock_backend.destroyed) == 2


# --- Trial lifecycle ---


@pytest.mark.asyncio
async def test_create_trial(started_manager, mock_backend):
    session = await started_manager.create_session("test", {})
    trial = await started_manager.create_trial(session.session_id, "", {})

    assert trial.trial_id in started_manager.trials
    assert trial.handle is not None
    assert trial.handle.workspace.is_dir()
    assert trial.profile_name == "test"
    assert trial.state == TrialState.READY
    assert mock_backend.created == [trial.trial_id]


def _task_root(tmp_path, setup="", verify="def verify(answer):\n    return answer == '42'\n"):
    task = tmp_path / "tasks" / "t1"
    (task / "workspace" / "src").mkdir(parents=True)
    (task / "workspace" / "src" / "main.py").write_text("print('hi')\n")
    (task / "verify.py").write_text(verify)
    if setup:
        (task / "task.yaml").write_text(setup)
    return tmp_path / "tasks"


@pytest.mark.asyncio
async def test_create_trial_copies_task_workspace(workspace_root, mock_backend, tmp_path):
    profile = SandboxProfile(name="test", backend_type="mock", task_root=_task_root(tmp_path))
    mgr = SandboxManager({"test": profile}, {"mock": mock_backend}, workspace_root)
    await mgr.start()
    try:
        session = await mgr.create_session("test", {})
        trial = await mgr.create_trial(session.session_id, "t1", {})
        assert trial.task_id == "t1"
        assert (trial.workspace / "src" / "main.py").is_file()
        # Hidden task files never reach the workspace.
        assert not (trial.workspace / "verify.py").exists()
    finally:
        await mgr.close()


@pytest.mark.asyncio
async def test_create_trial_runs_setup_commands(workspace_root, mock_backend, tmp_path):
    root = _task_root(tmp_path, setup="setup_commands:\n  - pip install x\n")
    profile = SandboxProfile(name="test", backend_type="mock", task_root=root)
    mgr = SandboxManager({"test": profile}, {"mock": mock_backend}, workspace_root)
    await mgr.start()
    try:
        session = await mgr.create_session("test", {})
        trial = await mgr.create_trial(session.session_id, "t1", {})
        assert mock_backend.executed == [(trial.trial_id, ["/bin/bash", "-lc", "pip install x"])]
    finally:
        await mgr.close()


@pytest.mark.asyncio
async def test_failed_setup_rolls_back(workspace_root, tmp_path):
    backend = MockBackend(ExecutionResult(1, "", "no network"))
    root = _task_root(tmp_path, setup="setup_commands:\n  - pip install x\n")
    profile = SandboxProfile(name="test", backend_type="mock", task_root=root)
    mgr = SandboxManager({"test": profile}, {"mock": backend}, workspace_root)
    await mgr.start()
    try:
        session = await mgr.create_session("test", {})
        with pytest.raises(ValueError, match="setup command failed"):
            await mgr.create_trial(session.session_id, "t1", {})
        assert backend.destroyed == backend.created
        assert mgr.runtimes["test"].live_trials == 0
        assert not mgr.trials
    finally:
        await mgr.close()


@pytest.mark.asyncio
async def test_unknown_task_rejected(workspace_root, mock_backend, tmp_path):
    profile = SandboxProfile(name="test", backend_type="mock", task_root=_task_root(tmp_path))
    mgr = SandboxManager({"test": profile}, {"mock": mock_backend}, workspace_root)
    await mgr.start()
    try:
        session = await mgr.create_session("test", {})
        with pytest.raises(ValueError, match="unknown task"):
            await mgr.create_trial(session.session_id, "missing", {})
        assert mgr.runtimes["test"].live_trials == 0
    finally:
        await mgr.close()


@pytest.mark.asyncio
async def test_compute_reward_runs_verify(workspace_root, mock_backend, tmp_path):
    profile = SandboxProfile(name="test", backend_type="mock", task_root=_task_root(tmp_path))
    mgr = SandboxManager({"test": profile}, {"mock": mock_backend}, workspace_root)
    await mgr.start()
    try:
        session = await mgr.create_session("test", {})
        trial = await mgr.create_trial(session.session_id, "t1", {})
        assert (await mgr.compute_reward(trial.trial_id, answer="42")).passed
        assert not (await mgr.compute_reward(trial.trial_id, answer="41")).passed
    finally:
        await mgr.close()


@pytest.mark.asyncio
async def test_create_trial_max_capacity(started_manager):
    session = await started_manager.create_session("test", {})
    for i in range(5):
        await started_manager.create_trial(session.session_id, "", {})

    with pytest.raises(RuntimeError, match="capacity exhausted"):
        await started_manager.create_trial(session.session_id, "", {})


@pytest.mark.asyncio
async def test_create_trial_cleanup_on_backend_failure(workspace_root):
    profile = SandboxProfile(name="test", backend_type="mock", max_live_trials=5)
    backend = FailingCreateBackend()
    mgr = SandboxManager({"test": profile}, {"mock": backend}, workspace_root)
    await mgr.start()

    session = await mgr.create_session("test", {})
    with pytest.raises(RuntimeError, match="backend create failed"):
        await mgr.create_trial(session.session_id, "", {})

    # live_trials count should be rolled back
    assert mgr.runtimes["test"].live_trials == 0
    await mgr.close()


@pytest.mark.asyncio
async def test_destroy_trial(started_manager, mock_backend):
    session = await started_manager.create_session("test", {})
    trial = await started_manager.create_trial(session.session_id, "", {})

    await started_manager.destroy_trial(trial.trial_id)

    assert trial.trial_id not in started_manager.trials
    assert not trial.handle.workspace.exists()
    assert mock_backend.destroyed == [trial.trial_id]
    assert started_manager.runtimes["test"].live_trials == 0


@pytest.mark.asyncio
async def test_destroy_trial_idempotent(started_manager):
    session = await started_manager.create_session("test", {})
    trial = await started_manager.create_trial(session.session_id, "", {})

    await started_manager.destroy_trial(trial.trial_id)
    await started_manager.destroy_trial(trial.trial_id)  # should not raise


# --- Action execution ---


@pytest.mark.asyncio
async def test_execute_action(started_manager, mock_backend):
    mock_backend._execute_result = ExecutionResult(0, "hello\n", "")
    session = await started_manager.create_session("test", {})
    trial = await started_manager.create_trial(session.session_id, "", {})

    resp = await started_manager.execute_action(trial.trial_id, "bash", {"command": "echo hello"}, timeout=10)
    assert resp.success is True
    assert resp.execution_seconds > 0


@pytest.mark.asyncio
async def test_execute_action_error_response(started_manager):
    session = await started_manager.create_session("test", {})
    trial = await started_manager.create_trial(session.session_id, "", {})

    resp = await started_manager.execute_action(trial.trial_id, "bash", {"command": "fail"}, timeout=10)
    # MockBackend returns rc=0 empty stdout, but bash dispatch will fail
    # because the _run method raises on non-zero OR empty stdout with the
    # default mock result. Actually the default mock returns rc=0, stdout="",
    # which means bash tool will succeed but return "". Let me test disabled tool.
    resp = await started_manager.execute_action(trial.trial_id, "query_db", {"db_name": "x", "query": "y"}, timeout=10)
    assert resp.success is False
    assert "not enabled" in resp.error


@pytest.mark.asyncio
async def test_execute_action_on_destroyed_trial(started_manager):
    session = await started_manager.create_session("test", {})
    trial = await started_manager.create_trial(session.session_id, "", {})
    await started_manager.destroy_trial(trial.trial_id)

    with pytest.raises(KeyError, match="unknown trial"):
        await started_manager.execute_action(trial.trial_id, "bash", {"command": "echo"}, timeout=10)


@pytest.mark.asyncio
async def test_execute_action_return_answer(started_manager):
    session = await started_manager.create_session("test", {})
    trial = await started_manager.create_trial(session.session_id, "", {})

    resp = await started_manager.execute_action(trial.trial_id, "return_answer", {"answer": "42"}, timeout=10)
    assert resp.success is True
    assert resp.result["answer"] == "42"


# --- Heartbeat ---


@pytest.mark.asyncio
async def test_heartbeat(started_manager):
    session = await started_manager.create_session("test", {})
    trial = await started_manager.create_trial(session.session_id, "", {})
    old_deadline = trial.lease_deadline

    await asyncio.sleep(0.01)
    updated = await started_manager.heartbeat(trial.trial_id)
    assert updated.lease_deadline > old_deadline


# --- Capacity ---


@pytest.mark.asyncio
async def test_capacity(started_manager):
    session = await started_manager.create_session("test", {})
    await started_manager.create_trial(session.session_id, "", {})

    caps = started_manager.capacity()
    assert len(caps) == 1
    assert caps[0].profile == "test"
    assert caps[0].live_trials == 1
    assert caps[0].max_live_trials == 5


# --- Lease reaper ---


@pytest.mark.asyncio
async def test_lease_reaper(workspace_root, mock_backend):
    profile = SandboxProfile(
        name="test",
        backend_type="mock",
        max_live_trials=10,
        lease_seconds=0.1,
    )
    mgr = SandboxManager(
        {"test": profile},
        {"mock": mock_backend},
        workspace_root,
        lease_check_interval=0.1,
    )
    await mgr.start()
    try:
        session = await mgr.create_session("test", {})
        trial = await mgr.create_trial(session.session_id, "", {})
        assert trial.trial_id in mgr.trials

        await asyncio.sleep(0.4)

        assert trial.trial_id not in mgr.trials
        assert mock_backend.destroyed == [trial.trial_id]
    finally:
        await mgr.close()


# --- Evaluate ---


SLOW_VERIFY = "import time\ndef verify(answer):\n    time.sleep(1)\n    return 1\n"


@pytest.mark.asyncio
async def test_compute_reward_uses_profile_timeout(workspace_root, mock_backend, tmp_path):
    profile = SandboxProfile(
        name="test",
        backend_type="mock",
        max_live_trials=5,
        reward_timeout_seconds=0.01,
        task_root=_task_root(tmp_path, verify=SLOW_VERIFY),
    )
    mgr = SandboxManager({"test": profile}, {"mock": mock_backend}, workspace_root)
    await mgr.start()
    try:
        session = await mgr.create_session("test", {})
        trial = await mgr.create_trial(session.session_id, "t1", {})

        result = await mgr.compute_reward(trial.trial_id, answer="42")

        assert result.reward == 0.0
        assert result.passed is False
        assert result.details["error"] == "reward_timeout"
        assert result.details["timeout_seconds"] == 0.01
        assert trial.state == TrialState.READY
    finally:
        await mgr.close()


@pytest.mark.asyncio
async def test_compute_reward_timeout_override(workspace_root, mock_backend, tmp_path):
    profile = SandboxProfile(
        name="test",
        backend_type="mock",
        max_live_trials=5,
        reward_timeout_seconds=60.0,
        task_root=_task_root(tmp_path, verify=SLOW_VERIFY),
    )
    mgr = SandboxManager({"test": profile}, {"mock": mock_backend}, workspace_root)
    await mgr.start()
    try:
        session = await mgr.create_session("test", {})
        trial = await mgr.create_trial(session.session_id, "t1", {})

        result = await mgr.compute_reward(trial.trial_id, answer="42", timeout=0.01)

        assert result.details["error"] == "reward_timeout"
        assert result.details["timeout_seconds"] == 0.01
    finally:
        await mgr.close()


@pytest.mark.asyncio
async def test_compute_reward_not_implemented(started_manager):
    session = await started_manager.create_session("test", {})
    trial = await started_manager.create_trial(session.session_id, "", {})

    with pytest.raises(NotImplementedError):
        await started_manager.compute_reward(trial.trial_id, answer="42")
