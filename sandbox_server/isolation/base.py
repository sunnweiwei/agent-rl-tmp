import asyncio
import os
import signal
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sandbox_server.profiles import SandboxProfile


class IsolationUnavailable(RuntimeError):
    pass


@dataclass
class TrialHandle:
    """Opaque handle returned by create_trial. Backends may subclass."""

    trial_id: str
    workspace: Path


@dataclass(frozen=True)
class ExecutionResult:
    returncode: int
    stdout: str
    stderr: str


def _decode_output(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _kill_process_group_sync(process: subprocess.Popen, cleanup_timeout: float) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=cleanup_timeout)
    except subprocess.TimeoutExpired:
        pass


def _read_temp_output(handle, max_output_bytes: int) -> tuple[bytes, bool]:
    handle.seek(0, os.SEEK_END)
    size = handle.tell()
    handle.seek(0)
    return handle.read(max_output_bytes), size > max_output_bytes


def collect_process_output_sync(
    argv: list[str],
    timeout: float,
    max_output_bytes: int,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    cleanup_timeout: float = 2.0,
) -> ExecutionResult:
    """Run a child process without touching the asyncio event loop.

    The sandbox server may run hundreds of execute_python actions in parallel.
    Keeping subprocess wait/read operations in worker threads prevents one slow
    or kernel-blocked jail from stalling uvicorn's main event loop and control
    endpoints such as /v1/capacity.
    """
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _kill_process_group_sync(process, cleanup_timeout)
            raise TimeoutError(f"process timed out after {timeout:g}s") from exc

        stdout, stdout_overflow = _read_temp_output(stdout_file, max_output_bytes)
        stderr, stderr_overflow = _read_temp_output(stderr_file, max_output_bytes)
        overflow = stdout_overflow or stderr_overflow
        stderr_text = _decode_output(stderr)
        if overflow:
            stderr_text += f"\noutput exceeded {max_output_bytes} bytes"
        returncode = process.returncode or 0
        if overflow and returncode == 0:
            returncode = 125
        return ExecutionResult(
            returncode=returncode,
            stdout=_decode_output(stdout),
            stderr=stderr_text,
        )


async def run_process_in_thread(
    argv: list[str],
    timeout: float,
    max_output_bytes: int,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> ExecutionResult:
    return await asyncio.to_thread(
        collect_process_output_sync,
        argv,
        timeout,
        max_output_bytes,
        cwd=cwd,
        env=env,
    )


async def collect_process_output(
    process: asyncio.subprocess.Process,
    timeout: float,
    max_output_bytes: int,
) -> ExecutionResult:
    overflow = asyncio.Event()
    cleanup_timeout = 2.0

    async def read_stream(stream: asyncio.StreamReader | None) -> bytes:
        if stream is None:
            return b""
        chunks: list[bytes] = []
        stored = 0
        while True:
            chunk = await stream.read(64 * 1024)
            if not chunk:
                break
            remaining = max_output_bytes - stored
            if remaining > 0:
                chunks.append(chunk[:remaining])
                stored += min(len(chunk), remaining)
            if len(chunk) > remaining:
                overflow.set()
        return b"".join(chunks)

    async def kill_and_wait() -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        # Some jailed children can get stuck in uninterruptible kernel sleep.
        # Never let cleanup hold the server action lock forever.
        try:
            await asyncio.wait_for(process.wait(), timeout=cleanup_timeout)
        except asyncio.TimeoutError:
            pass

    async def cancel_reader(task: asyncio.Task) -> None:
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=cleanup_timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    stdout_task = asyncio.create_task(read_stream(process.stdout))
    stderr_task = asyncio.create_task(read_stream(process.stderr))
    wait_task = asyncio.create_task(process.wait())
    overflow_task = asyncio.create_task(overflow.wait())
    try:
        done, _ = await asyncio.wait(
            {wait_task, overflow_task},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            raise asyncio.TimeoutError
        if overflow_task in done and overflow.is_set() and not wait_task.done():
            await kill_and_wait()
            if not wait_task.done():
                raise asyncio.TimeoutError
        elif wait_task not in done:
            await wait_task
        stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        await kill_and_wait()
        wait_task.cancel()
        await asyncio.gather(wait_task, return_exceptions=True)
        await asyncio.gather(
            cancel_reader(stdout_task),
            cancel_reader(stderr_task),
            return_exceptions=True,
        )
        raise
    finally:
        overflow_task.cancel()
        await asyncio.gather(overflow_task, return_exceptions=True)

    stderr_text = stderr.decode("utf-8", errors="replace")
    if overflow.is_set():
        stderr_text += f"\noutput exceeded {max_output_bytes} bytes"
    returncode = process.returncode or 0
    if overflow.is_set() and returncode == 0:
        returncode = 125
    return ExecutionResult(
        returncode=returncode,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr_text,
    )


class IsolationBackend(ABC):
    """Per-trial isolation backend.

    One-shot backends (nsjail, subprocess): create_trial is minimal,
    each execute spawns a fresh isolated process.

    Persistent backends (Docker): create_trial starts a container,
    execute runs commands inside it via docker exec.
    """

    name: str

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    async def create_trial(
        self,
        trial_id: str,
        profile: SandboxProfile,
        workspace: Path,
        task_config: dict[str, Any] | None = None,
    ) -> TrialHandle: ...

    @abstractmethod
    async def execute(
        self,
        handle: TrialHandle,
        profile: SandboxProfile,
        argv: list[str],
        timeout: float,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult: ...

    @abstractmethod
    async def destroy(self, handle: TrialHandle) -> None: ...
