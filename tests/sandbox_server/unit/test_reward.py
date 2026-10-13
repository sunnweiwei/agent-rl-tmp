import pytest

from sandbox_server.isolation.base import ExecutionResult, TrialHandle
from sandbox_server.profiles import SandboxProfile
from sandbox_server.reward import compute_task_reward
from sandbox_server.tasks import SCRATCH_TASK, resolve_task
from tests.sandbox_server.conftest import MockBackend


def _task(tmp_path, verify_src=None, task_yaml=None, tests=False):
    task_dir = tmp_path / "tasks" / "t"
    task_dir.mkdir(parents=True)
    if verify_src is not None:
        (task_dir / "verify.py").write_text(verify_src)
    if task_yaml is not None:
        (task_dir / "task.yaml").write_text(task_yaml)
    if tests:
        (task_dir / "tests").mkdir()
        (task_dir / "tests" / "test_x.py").write_text("def test_x(): pass\n")
    return resolve_task(tmp_path / "tasks", "t")


@pytest.fixture
def handle(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return TrialHandle(trial_id="trial", workspace=workspace)


PROFILE = SandboxProfile(name="test", backend_type="mock")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "reward", "passed", "reason"),
    [
        ("return answer == '4'", 1.0, True, ""),
        ("return answer == '5'", 0.0, False, ""),
        ("return 0.5", 0.5, False, ""),
        ("return (1, 'exact')", 1.0, True, "exact"),
        ("return {'score': 0.3, 'passed': True, 'reason': 'partial'}", 0.3, True, "partial"),
    ],
)
async def test_verify_return_types(tmp_path, handle, body, reward, passed, reason):
    task = _task(tmp_path, f"def verify(answer):\n    {body}\n")
    result = await compute_task_reward(MockBackend(), handle, PROFILE, task, "4")
    assert (result.reward, result.passed, result.details["reason"]) == (reward, passed, reason)


@pytest.mark.asyncio
async def test_verify_receives_workspace(tmp_path, handle):
    (handle.workspace / "out.txt").write_text("done")
    task = _task(
        tmp_path,
        "import os\n"
        "def verify(answer, workspace):\n"
        "    return open(os.path.join(workspace, 'out.txt')).read() == 'done'\n",
    )
    result = await compute_task_reward(MockBackend(), handle, PROFILE, task, "")
    assert result.passed


@pytest.mark.asyncio
async def test_verify_without_answer(tmp_path, handle):
    task = _task(tmp_path, "def verify(answer):\n    return 1\n")
    result = await compute_task_reward(MockBackend(), handle, PROFILE, task, None)
    assert result.reward == 0.0
    assert result.details["reason"] == "no answer provided"


@pytest.mark.asyncio
async def test_command_reward_stages_hidden_tests(tmp_path, handle):
    task = _task(tmp_path, task_yaml="reward:\n  type: command\n  command: pytest -q .eval/tests\n", tests=True)
    backend = MockBackend(ExecutionResult(0, "1 passed", ""))
    result = await compute_task_reward(backend, handle, PROFILE, task, None)
    assert result.passed and result.reward == 1.0
    assert (handle.workspace / ".eval" / "tests" / "test_x.py").is_file()
    assert backend.executed == [("trial", ["/bin/bash", "-lc", "pytest -q .eval/tests"])]


@pytest.mark.asyncio
async def test_command_reward_failure(tmp_path, handle):
    task = _task(tmp_path, task_yaml="reward:\n  type: command\n  command: 'false'\n")
    result = await compute_task_reward(MockBackend(ExecutionResult(1, "", "boom")), handle, PROFILE, task, None)
    assert not result.passed
    assert result.details["returncode"] == 1
    assert result.details["stderr"] == "boom"


@pytest.mark.asyncio
async def test_no_reward(handle):
    with pytest.raises(NotImplementedError):
        await compute_task_reward(MockBackend(), handle, PROFILE, SCRATCH_TASK, "x")
