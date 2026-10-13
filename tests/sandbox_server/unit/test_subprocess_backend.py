import sys
from pathlib import Path

import pytest

from sandbox_server.isolation.base import TrialHandle
from sandbox_server.isolation.subprocess import SubprocessBackend
from sandbox_server.profiles import SandboxProfile


@pytest.fixture
def backend():
    return SubprocessBackend()


@pytest.fixture
def trusted():
    return SandboxProfile(name="test", backend_type="subprocess", trusted_subprocess=True)


@pytest.fixture
def untrusted():
    return SandboxProfile(name="test", backend_type="subprocess", trusted_subprocess=False)


def test_available(backend):
    assert backend.available is True


@pytest.mark.asyncio
async def test_create_trial_untrusted(backend, untrusted):
    with pytest.raises(PermissionError):
        await backend.create_trial("t1", untrusted, Path("/tmp/ws"))


@pytest.mark.asyncio
async def test_create_trial_trusted(backend, trusted):
    handle = await backend.create_trial("t1", trusted, Path("/tmp/ws"))
    assert handle.trial_id == "t1"


@pytest.mark.asyncio
async def test_execute_untrusted(backend, untrusted):
    handle = TrialHandle(trial_id="t1", workspace=Path("/tmp"))
    with pytest.raises(PermissionError):
        await backend.execute(handle, untrusted, ["echo", "hi"], timeout=5)


@pytest.mark.asyncio
async def test_execute_trusted(backend, trusted, tmp_path):
    handle = TrialHandle(trial_id="t1", workspace=tmp_path)
    result = await backend.execute(handle, trusted, [sys.executable, "-c", "print('hello')"], timeout=5)
    assert result.returncode == 0
    assert "hello" in result.stdout


@pytest.mark.asyncio
async def test_execute_captures_returncode(backend, trusted, tmp_path):
    handle = TrialHandle(trial_id="t1", workspace=tmp_path)
    result = await backend.execute(handle, trusted, [sys.executable, "-c", "exit(7)"], timeout=5)
    assert result.returncode == 7


@pytest.mark.asyncio
async def test_destroy_noop(backend):
    handle = TrialHandle(trial_id="t1", workspace=Path("/tmp"))
    await backend.destroy(handle)
