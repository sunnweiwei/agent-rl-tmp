import os
from pathlib import Path
from typing import Any

from sandbox_server.isolation.base import (
    ExecutionResult,
    IsolationBackend,
    TrialHandle,
    run_process_in_thread,
)
from sandbox_server.profiles import SandboxProfile


class SubprocessBackend(IsolationBackend):
    """Unsandboxed subprocess execution. Dev/test only — fail-closed for
    profiles that do not explicitly set trusted_subprocess=True."""

    name = "subprocess"

    def __init__(self, max_output_bytes: int = 1_000_000):
        self.max_output_bytes = max_output_bytes

    @property
    def available(self) -> bool:
        return True

    async def create_trial(
        self,
        trial_id: str,
        profile: SandboxProfile,
        workspace: Path,
        task_config: dict[str, Any] | None = None,
    ) -> TrialHandle:
        if not profile.trusted_subprocess:
            raise PermissionError(f"profile {profile.name!r} cannot run with the subprocess backend")
        return TrialHandle(trial_id=trial_id, workspace=workspace)

    async def execute(
        self,
        handle: TrialHandle,
        profile: SandboxProfile,
        argv: list[str],
        timeout: float,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        if not profile.trusted_subprocess:
            raise PermissionError(f"profile {profile.name!r} cannot run with the subprocess backend")
        return await run_process_in_thread(
            argv,
            timeout,
            self.max_output_bytes,
            cwd=handle.workspace,
            env={**os.environ, **(env or {})},
        )

    async def destroy(self, handle: TrialHandle) -> None:
        pass
