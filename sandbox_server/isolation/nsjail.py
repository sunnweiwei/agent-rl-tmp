import os
import shutil
from pathlib import Path
from typing import Any

from sandbox_server.isolation.base import (
    ExecutionResult,
    IsolationBackend,
    IsolationUnavailable,
    TrialHandle,
    run_process_in_thread,
)
from sandbox_server.profiles import SandboxProfile


class NsjailBackend(IsolationBackend):
    """One-shot nsjail jails: every action runs in a fresh jail over the
    trial workspace, mounted at /workspace."""

    name = "nsjail"

    def __init__(self, executable: str = "nsjail", max_output_bytes: int = 1_000_000):
        self.executable = executable
        self.max_output_bytes = max_output_bytes

    @property
    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    async def create_trial(
        self,
        trial_id: str,
        profile: SandboxProfile,
        workspace: Path,
        task_config: dict[str, Any] | None = None,
    ) -> TrialHandle:
        return TrialHandle(trial_id=trial_id, workspace=workspace)

    async def execute(
        self,
        handle: TrialHandle,
        profile: SandboxProfile,
        argv: list[str],
        timeout: float,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        command = self._build_command(profile, handle.workspace, argv, timeout)
        # timeout+2: give nsjail's --time_limit a chance to fire first;
        # our timeout is the fallback if nsjail's enforcement fails.
        return await run_process_in_thread(
            command,
            timeout + 2,
            self.max_output_bytes,
            env={**os.environ, **(env or {})},
        )

    async def destroy(self, handle: TrialHandle) -> None:
        pass

    def _build_command(
        self,
        profile: SandboxProfile,
        workspace: Path,
        argv: list[str],
        timeout: float,
    ) -> list[str]:
        if not self.available:
            raise IsolationUnavailable(f"{self.executable!r} is not installed")
        if profile.nsjail_config is None or not profile.nsjail_config.is_file():
            raise IsolationUnavailable(f"profile {profile.name!r} has no readable nsjail config")
        if profile.rootfs is None or not profile.rootfs.is_dir():
            raise IsolationUnavailable(f"profile {profile.name!r} has no readable rootfs")
        return [
            self.executable,
            # Suppress nsjail's stderr boilerplate: the INFO preamble plus the
            # constant UID/EUID=0 warnings from our uidmap config. stderr is
            # surfaced to the agent on failures — the traceback should be the
            # first thing it reads. FATAL errors still print.
            "--really_quiet",
            "--config",
            str(profile.nsjail_config),
            "--chroot",
            str(profile.rootfs),
            "--bindmount",
            f"{workspace}:/workspace",
            "--cwd",
            "/workspace",
            "--time_limit",
            str(max(1, int(timeout))),
            "--",
            *argv,
        ]
