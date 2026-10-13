import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sandbox_server.isolation.base import (
    ExecutionResult,
    IsolationBackend,
    IsolationUnavailable,
    TrialHandle,
    collect_process_output,
)
from sandbox_server.profiles import SandboxProfile


@dataclass
class DockerTrialHandle(TrialHandle):
    """Trial handle carrying Docker container state."""

    container_id: str = ""
    container_name: str = ""
    image: str = ""


class DockerBackend(IsolationBackend):
    """Per-trial Docker containers. The image comes from ``task.yaml``
    (``image``) or the server default.

    Lifecycle:
      create_trial  -> docker create + docker start
      execute       -> docker exec
      destroy       -> docker rm -f
    """

    name = "docker"

    def __init__(
        self,
        docker_executable: str = "docker",
        default_image: str = "python:3.12-slim",
        mem_limit: str = "8g",
        cpu_limit: float = 2.0,
        pids_limit: int = 512,
        tmpfs_size: str = "512m",
        max_output_bytes: int = 1_000_000,
        network: str = "none",
    ):
        self.docker_executable = docker_executable
        self.default_image = default_image
        self.mem_limit = mem_limit
        self.cpu_limit = cpu_limit
        self.pids_limit = pids_limit
        self.tmpfs_size = tmpfs_size
        self.max_output_bytes = max_output_bytes
        # "none" = isolated (safe default).
        self.network = network

    @property
    def available(self) -> bool:
        return shutil.which(self.docker_executable) is not None

    async def create_trial(
        self,
        trial_id: str,
        profile: SandboxProfile,
        workspace: Path,
        task_config: dict[str, Any] | None = None,
    ) -> DockerTrialHandle:
        if not self.available:
            raise IsolationUnavailable(f"{self.docker_executable!r} is not installed")

        cfg = task_config or {}
        image = cfg.get("image", self.default_image)
        container_name = f"sandbox-trial-{trial_id[:16]}"
        user = cfg.get("user", "1000:1000")

        # Hardened container: read-only rootfs, no capabilities, no privilege
        # escalation, network per config (default none). Agent code runs
        # inside; it cannot reach the Docker socket or host filesystem.
        # "sleep infinity" keeps the container alive — we use docker exec for actions.
        create_cmd = [
            self.docker_executable,
            "create",
            "--name",
            container_name,
            "--network",
            self.network,
            "--read-only",
            "--user",
            user,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            self.mem_limit,
            "--cpus",
            str(self.cpu_limit),
            "--pids-limit",
            str(self.pids_limit),
            "--tmpfs",
            f"/tmp:size={self.tmpfs_size}",
        ]
        # extra_tmpfs: writable overlay dirs on a read-only rootfs.
        for extra_path in cfg.get("extra_tmpfs", []):
            create_cmd.extend(["--tmpfs", f"{extra_path}:size=100m"])
        # The workspace is always /workspace (same as nsjail) and is the cwd.
        create_cmd.extend(
            [
                "-v",
                f"{workspace}:/workspace:rw",
                "--workdir",
                "/workspace",
                image,
                "sleep",
                "infinity",
            ]
        )

        proc = await asyncio.create_subprocess_exec(
            *create_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"docker create failed: {stderr.decode('utf-8', errors='replace').strip()}")
        container_id = stdout.decode().strip()

        start_proc = await asyncio.create_subprocess_exec(
            self.docker_executable,
            "start",
            container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, start_err = await start_proc.communicate()
        if start_proc.returncode != 0:
            await self._force_remove(container_id)
            raise RuntimeError(f"docker start failed: {start_err.decode('utf-8', errors='replace').strip()}")

        return DockerTrialHandle(
            trial_id=trial_id,
            workspace=workspace,
            container_id=container_id,
            container_name=container_name,
            image=image,
        )

    async def execute(
        self,
        handle: TrialHandle,
        profile: SandboxProfile,
        argv: list[str],
        timeout: float,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        if not isinstance(handle, DockerTrialHandle):
            raise TypeError("DockerBackend requires a DockerTrialHandle")

        cmd = [self.docker_executable, "exec"]
        if env:
            for k, v in env.items():
                cmd.extend(["-e", f"{k}={v}"])
        cmd.extend([handle.container_id, *argv])

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return await collect_process_output(process, timeout, self.max_output_bytes)

    async def destroy(self, handle: TrialHandle) -> None:
        if not isinstance(handle, DockerTrialHandle):
            return
        if handle.container_id:
            await self._force_remove(handle.container_id)

    async def _force_remove(self, container_id: str) -> None:
        # docker rm -f kills all processes and removes the container.
        # This is the only cleanup needed.
        proc = await asyncio.create_subprocess_exec(
            self.docker_executable,
            "rm",
            "-f",
            container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
