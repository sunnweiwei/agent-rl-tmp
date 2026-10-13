"""Task directory resolution.

A task is a directory under the daemon's task root::

    <task_root>/<task_id>/
        task.yaml     optional; image, setup_commands, reward settings
        workspace/    optional; copied into the trial workspace at creation
        tests/        optional; hidden from the agent, staged only for reward
        verify.py     optional; trusted ``verify(answer, workspace)`` reward

Clients only send ``task_id``. Every path is resolved on the daemon so rollout
workers cannot point the sandbox at arbitrary host files.
"""

import copy
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


class TaskError(ValueError):
    pass


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_dir: Path | None
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def workspace_dir(self) -> Path | None:
        if self.task_dir is None:
            return None
        path = self.task_dir / "workspace"
        return path if path.is_dir() else None

    @property
    def tests_dir(self) -> Path | None:
        if self.task_dir is None:
            return None
        path = self.task_dir / "tests"
        return path if path.is_dir() else None

    @property
    def verify_py(self) -> Path | None:
        if self.task_dir is None:
            return None
        path = self.task_dir / "verify.py"
        return path if path.is_file() else None

    @property
    def setup_commands(self) -> list[str]:
        return [str(cmd) for cmd in self.config.get("setup_commands", [])]

    @property
    def reward_config(self) -> dict[str, Any]:
        reward = dict(self.config.get("reward") or {})
        reward.setdefault("type", "verify" if self.verify_py is not None else "none")
        return reward


SCRATCH_TASK = TaskSpec(task_id="", task_dir=None)


def resolve_task(task_root: Path | None, task_id: str) -> TaskSpec:
    if not task_id:
        return SCRATCH_TASK
    if task_root is None:
        raise TaskError("SANDBOX_TASK_ROOT is not configured on the sandbox daemon")
    task_dir, config = _resolve_cached(str(task_root.resolve()), task_id)
    return TaskSpec(task_id=task_id, task_dir=task_dir, config=copy.deepcopy(config))


@lru_cache(maxsize=16384)
def _resolve_cached(root_str: str, task_id: str) -> tuple[Path, dict[str, Any]]:
    root = Path(root_str)
    relative = Path(task_id)
    if relative.is_absolute():
        raise TaskError("task_id must be relative")
    task_dir = (root / relative).resolve()
    try:
        task_dir.relative_to(root)
    except ValueError as exc:
        raise TaskError("task_id escapes task root") from exc
    if not task_dir.is_dir():
        raise TaskError(f"unknown task: {task_id!r}")

    config: dict[str, Any] = {}
    config_path = task_dir / "task.yaml"
    if config_path.is_file():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(config, dict):
            raise TaskError(f"invalid task config: {config_path}")
    return task_dir, config
