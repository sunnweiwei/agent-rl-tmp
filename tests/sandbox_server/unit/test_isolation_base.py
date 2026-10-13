import asyncio
import sys
from pathlib import Path

import pytest

from sandbox_server.isolation.base import (
    ExecutionResult,
    IsolationBackend,
    TrialHandle,
    collect_process_output,
)


def test_trial_handle():
    h = TrialHandle(trial_id="abc", workspace=Path("/tmp/ws"))
    assert h.trial_id == "abc"
    assert h.workspace == Path("/tmp/ws")


def test_execution_result_frozen():
    r = ExecutionResult(returncode=0, stdout="ok", stderr="")
    with pytest.raises(AttributeError):
        r.returncode = 1


def test_isolation_backend_abstract():
    with pytest.raises(TypeError):
        IsolationBackend()


@pytest.mark.asyncio
async def test_collect_output_captures():
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "print('hello')",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    result = await collect_process_output(proc, timeout=5.0, max_output_bytes=1_000_000)
    assert result.returncode == 0
    assert "hello" in result.stdout


@pytest.mark.asyncio
async def test_collect_output_captures_stderr():
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('err\\n')",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    result = await collect_process_output(proc, timeout=5.0, max_output_bytes=1_000_000)
    assert "err" in result.stderr


@pytest.mark.asyncio
async def test_collect_output_nonzero_returncode():
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "exit(42)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    result = await collect_process_output(proc, timeout=5.0, max_output_bytes=1_000_000)
    assert result.returncode == 42


@pytest.mark.asyncio
async def test_collect_output_timeout():
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    with pytest.raises(asyncio.TimeoutError):
        await collect_process_output(proc, timeout=0.5, max_output_bytes=1_000_000)


@pytest.mark.asyncio
async def test_collect_output_overflow():
    # Slow-drip output so overflow is detected while process is still alive
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time\nfor i in range(200):\n print('x' * 100)\n time.sleep(0.01)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    result = await collect_process_output(proc, timeout=10.0, max_output_bytes=500)
    assert len(result.stdout) <= 500
    assert "exceeded" in result.stderr
