from pathlib import Path
from unittest.mock import patch

import pytest

from sandbox_server.isolation.base import IsolationUnavailable, TrialHandle
from sandbox_server.isolation.nsjail import NsjailBackend
from sandbox_server.profiles import SandboxProfile


@pytest.fixture
def backend():
    return NsjailBackend(executable="nsjail")


@pytest.fixture
def profile_with_rootfs(tmp_path):
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    cfg = tmp_path / "jail.cfg"
    cfg.write_text("name: test")
    return SandboxProfile(
        name="test",
        backend_type="nsjail",
        rootfs=rootfs,
        nsjail_config=cfg,
    )


def test_available_false(backend):
    with patch("shutil.which", return_value=None):
        assert backend.available is False


def test_available_true(backend):
    with patch("shutil.which", return_value="/usr/bin/nsjail"):
        assert backend.available is True


@pytest.mark.asyncio
async def test_create_trial(backend):
    ws = Path("/tmp/ws")
    profile = SandboxProfile(name="test")
    handle = await backend.create_trial("t1", profile, ws)
    assert isinstance(handle, TrialHandle)
    assert handle.trial_id == "t1"
    assert handle.workspace == ws


@pytest.mark.asyncio
async def test_destroy_is_noop(backend):
    handle = TrialHandle(trial_id="t1", workspace=Path("/tmp/ws"))
    await backend.destroy(handle)  # should not raise


def test_build_command_no_binary(backend):
    profile = SandboxProfile(name="test")
    with patch("shutil.which", return_value=None):
        with pytest.raises(IsolationUnavailable, match="not installed"):
            backend._build_command(profile, Path("/ws"), ["echo"], 10)


def test_build_command_no_rootfs(backend, tmp_path):
    cfg = tmp_path / "jail.cfg"
    cfg.write_text("name: test")
    profile = SandboxProfile(name="test", nsjail_config=cfg)
    with patch("shutil.which", return_value="/usr/bin/nsjail"):
        with pytest.raises(IsolationUnavailable, match="rootfs"):
            backend._build_command(profile, Path("/ws"), ["echo"], 10)


def test_build_command_no_config(backend, tmp_path):
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    profile = SandboxProfile(name="test", rootfs=rootfs)
    with patch("shutil.which", return_value="/usr/bin/nsjail"):
        with pytest.raises(IsolationUnavailable, match="nsjail config"):
            backend._build_command(profile, Path("/ws"), ["echo"], 10)


def test_build_command_valid(backend, profile_with_rootfs):
    with patch("shutil.which", return_value="/usr/bin/nsjail"):
        cmd = backend._build_command(profile_with_rootfs, Path("/ws"), ["python3", "script.py"], 30)
    assert cmd[0] == "nsjail"
    assert "--config" in cmd
    assert "--chroot" in cmd
    assert "--bindmount" in cmd
    assert "/ws:/workspace" in cmd
    assert "--time_limit" in cmd
    assert "30" in cmd
    assert "--" in cmd
    idx = cmd.index("--")
    assert cmd[idx + 1 :] == ["python3", "script.py"]
