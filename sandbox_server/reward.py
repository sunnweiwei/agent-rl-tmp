"""Trusted task rewards.

Two reward types, selected by ``reward.type`` in ``task.yaml``:

``verify`` (default when the task has ``verify.py``)
    Imports ``verify.py`` in the server process and calls
    ``verify(answer)`` or ``verify(answer, workspace)``. The function may
    return a bool, a number, a ``(score, reason)`` tuple, or a dict with
    ``score``/``passed``/``reason``. Agent code never sees ``verify.py``.

``command``
    Copies the task's hidden ``tests/`` directory into ``.eval/tests`` of the
    trial workspace, then runs ``reward.command`` inside the trial (same
    isolation as the agent's tools). Exit code 0 means reward 1.0.
"""

import asyncio
import importlib.util
import inspect
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sandbox_server.isolation.base import IsolationBackend, TrialHandle
from sandbox_server.profiles import SandboxProfile
from sandbox_server.tasks import TaskSpec

_IMPORT_LOCK = threading.Lock()


@dataclass
class RewardResult:
    reward: float
    passed: bool
    details: dict[str, Any]


async def compute_task_reward(
    backend: IsolationBackend,
    handle: TrialHandle,
    profile: SandboxProfile,
    task: TaskSpec,
    answer: str | None,
) -> RewardResult:
    reward_type = task.reward_config["type"]
    if reward_type == "verify":
        return await asyncio.to_thread(_run_verify, task, answer, handle.workspace)
    if reward_type == "command":
        return await _run_command(backend, handle, profile, task)
    raise NotImplementedError(f"task {task.task_id!r} has no reward (reward.type={reward_type!r})")


def _run_verify(task: TaskSpec, answer: str | None, workspace: Path) -> RewardResult:
    if task.verify_py is None:
        raise FileNotFoundError(f"verify.py not found for task {task.task_id!r}")
    if answer is None:
        return RewardResult(reward=0.0, passed=False, details={"reason": "no answer provided"})

    spec = importlib.util.spec_from_file_location(f"task_verify_{abs(hash(task.task_id))}", task.verify_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load verifier: {task.verify_py}")
    module = importlib.util.module_from_spec(spec)
    with _IMPORT_LOCK:
        spec.loader.exec_module(module)

    verify = module.verify
    if len(inspect.signature(verify).parameters) >= 2:
        result = verify(answer, str(workspace))
    else:
        result = verify(answer)
    return _normalize(result)


def _normalize(result: Any) -> RewardResult:
    reason = ""
    passed = None
    if isinstance(result, dict):
        score = float(result.get("score", float(bool(result.get("passed", False)))))
        passed = result.get("passed")
        reason = str(result.get("reason", ""))
    elif isinstance(result, tuple):
        score = float(result[0])
        reason = str(result[1]) if len(result) > 1 else ""
    else:
        score = float(result)
    return RewardResult(
        reward=score,
        passed=bool(score >= 1.0) if passed is None else bool(passed),
        details={"reason": reason},
    )


async def _run_command(
    backend: IsolationBackend,
    handle: TrialHandle,
    profile: SandboxProfile,
    task: TaskSpec,
) -> RewardResult:
    config = task.reward_config
    command = config.get("command")
    if not command:
        raise ValueError(f"task {task.task_id!r} reward.type=command requires reward.command")
    if task.tests_dir is not None:
        await asyncio.to_thread(
            shutil.copytree, task.tests_dir, handle.workspace / ".eval" / "tests", dirs_exist_ok=True
        )
    timeout = float(config.get("timeout", profile.reward_timeout_seconds))
    result = await backend.execute(handle, profile, ["/bin/bash", "-lc", str(command)], timeout)
    passed = result.returncode == 0
    return RewardResult(
        reward=1.0 if passed else 0.0,
        passed=passed,
        details={
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
        },
    )
