import pytest

from sandbox_server.tasks import SCRATCH_TASK, TaskError, resolve_task


@pytest.fixture
def task_root(tmp_path):
    root = tmp_path / "tasks"
    task = root / "suite" / "t1"
    (task / "workspace").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "verify.py").write_text("def verify(answer):\n    return answer == '4'\n")
    (task / "task.yaml").write_text("setup_commands:\n  - echo hi\nreward:\n  type: command\n  command: pytest\n")
    (root / "bare").mkdir()
    (root / "bare" / "verify.py").write_text("def verify(answer):\n    return 1\n")
    return root


def test_empty_task_id_is_scratch(task_root):
    assert resolve_task(task_root, "") is SCRATCH_TASK
    assert resolve_task(None, "") is SCRATCH_TASK
    assert SCRATCH_TASK.reward_config["type"] == "none"


def test_resolves_task_layout(task_root):
    task = resolve_task(task_root, "suite/t1")
    assert task.task_dir == (task_root / "suite" / "t1").resolve()
    assert task.workspace_dir is not None
    assert task.tests_dir is not None
    assert task.verify_py is not None
    assert task.setup_commands == ["echo hi"]
    assert task.reward_config == {"type": "command", "command": "pytest"}


def test_reward_defaults_to_verify_when_verify_py_exists(task_root):
    task = resolve_task(task_root, "bare")
    assert task.config == {}
    assert task.workspace_dir is None
    assert task.reward_config == {"type": "verify"}


def test_requires_task_root(task_root):
    with pytest.raises(TaskError, match="SANDBOX_TASK_ROOT"):
        resolve_task(None, "suite/t1")


@pytest.mark.parametrize("task_id", ["../outside", "/etc", "suite/../../x"])
def test_rejects_escape(task_root, task_id):
    with pytest.raises(TaskError):
        resolve_task(task_root, task_id)


def test_rejects_unknown_task(task_root):
    with pytest.raises(TaskError, match="unknown task"):
        resolve_task(task_root, "suite/missing")


def test_config_is_copied(task_root):
    resolve_task(task_root, "suite/t1").config["setup_commands"].append("mutated")
    assert resolve_task(task_root, "suite/t1").setup_commands == ["echo hi"]
