"""Docker backend tests — all Docker calls are mocked."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from sandbox_server.isolation.base import TrialHandle
from sandbox_server.isolation.docker import DockerBackend, DockerTrialHandle
from sandbox_server.profiles import SandboxProfile


@pytest.fixture
def backend():
    return DockerBackend(docker_executable="docker", default_image="test:latest")


@pytest.fixture
def profile():
    return SandboxProfile(name="ade", backend_type="docker")


def _mock_process(stdout=b"", stderr=b"", returncode=0):
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    proc.stdout = None
    proc.stderr = None
    proc.wait = AsyncMock(return_value=returncode)
    proc.pid = 12345
    return proc


def test_available_false(backend):
    with patch("shutil.which", return_value=None):
        assert backend.available is False


def test_available_true(backend):
    with patch("shutil.which", return_value="/usr/bin/docker"):
        assert backend.available is True


def test_docker_trial_handle():
    h = DockerTrialHandle(
        trial_id="t1",
        workspace=Path("/ws"),
        container_id="abc123",
        container_name="sandbox-trial-t1",
        image="test:latest",
    )
    assert h.container_id == "abc123"
    assert isinstance(h, TrialHandle)


@pytest.mark.asyncio
async def test_create_trial_success(backend, profile, tmp_path):
    create_proc = _mock_process(stdout=b"container_id_abc\n")
    start_proc = _mock_process()

    call_count = 0

    async def mock_exec(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return create_proc
        return start_proc

    with patch("shutil.which", return_value="/usr/bin/docker"):
        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec) as mock:
            handle = await backend.create_trial("t1", profile, tmp_path)

    assert isinstance(handle, DockerTrialHandle)
    assert handle.container_id == "container_id_abc"
    assert handle.image == "test:latest"

    first_call_args = mock.call_args_list[0][0]
    assert "docker" in first_call_args
    assert "create" in first_call_args
    assert "--network" in first_call_args
    assert "none" in first_call_args
    assert "--read-only" in first_call_args
    assert "--cap-drop" in first_call_args


@pytest.mark.asyncio
async def test_create_trial_custom_image(backend, profile, tmp_path):
    create_proc = _mock_process(stdout=b"cid\n")
    start_proc = _mock_process()
    call_count = 0

    async def mock_exec(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return create_proc if call_count == 1 else start_proc

    with patch("shutil.which", return_value="/usr/bin/docker"):
        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            handle = await backend.create_trial("t1", profile, tmp_path, task_config={"image": "custom:v2"})

    assert handle.image == "custom:v2"


@pytest.mark.asyncio
async def test_create_trial_bridge_network(profile, tmp_path):
    """ADE compose sets ADE_TRIAL_NETWORK=bridge for dbt deps access."""
    bridge_backend = DockerBackend(docker_executable="docker", default_image="test:latest", network="bridge")
    create_proc = _mock_process(stdout=b"cid\n")
    start_proc = _mock_process()
    call_count = 0

    async def mock_exec(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return create_proc if call_count == 1 else start_proc

    with patch("shutil.which", return_value="/usr/bin/docker"):
        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec) as mock:
            await bridge_backend.create_trial("t1", profile, tmp_path)

    first_call_args = mock.call_args_list[0][0]
    idx = list(first_call_args).index("--network")
    assert first_call_args[idx + 1] == "bridge"


@pytest.mark.asyncio
async def test_create_trial_extra_tmpfs(backend, profile, tmp_path):
    """ADE task_config adds extra_tmpfs for eval dirs (/tests, /scripts, etc.)."""
    create_proc = _mock_process(stdout=b"cid\n")
    start_proc = _mock_process()
    call_count = 0

    async def mock_exec(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return create_proc if call_count == 1 else start_proc

    with patch("shutil.which", return_value="/usr/bin/docker"):
        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec) as mock:
            await backend.create_trial(
                "t1",
                profile,
                tmp_path,
                task_config={"extra_tmpfs": ["/tests", "/scripts", "/seeds"]},
            )

    create_args = list(mock.call_args_list[0][0])
    # Each extra_tmpfs path should appear as --tmpfs /path:size=100m
    tmpfs_indices = [i for i, a in enumerate(create_args) if a == "--tmpfs"]
    tmpfs_values = [create_args[i + 1] for i in tmpfs_indices]
    assert "/tests:size=100m" in tmpfs_values
    assert "/scripts:size=100m" in tmpfs_values
    assert "/seeds:size=100m" in tmpfs_values


@pytest.mark.asyncio
async def test_create_trial_start_failure_rolls_back(backend, profile, tmp_path):
    create_proc = _mock_process(stdout=b"cid\n")
    start_proc = _mock_process(stderr=b"start failed", returncode=1)
    rm_proc = _mock_process()
    call_count = 0

    async def mock_exec(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return create_proc
        if call_count == 2:
            return start_proc
        return rm_proc  # docker rm -f cleanup

    with patch("shutil.which", return_value="/usr/bin/docker"):
        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec) as mock:
            with pytest.raises(RuntimeError, match="start failed"):
                await backend.create_trial("t1", profile, tmp_path)

    # Verify cleanup was called (docker rm -f)
    rm_call = mock.call_args_list[2][0]
    assert "rm" in rm_call
    assert "-f" in rm_call


@pytest.mark.asyncio
async def test_execute(backend, profile):
    handle = DockerTrialHandle(
        trial_id="t1",
        workspace=Path("/ws"),
        container_id="cid123",
    )
    exec_proc = _mock_process(stdout=b"output\n")

    with patch("asyncio.create_subprocess_exec", return_value=exec_proc) as mock:
        await backend.execute(handle, profile, ["echo", "hi"], timeout=5)

    call_args = mock.call_args[0]
    assert "docker" in call_args
    assert "exec" in call_args
    assert "cid123" in call_args
    assert "echo" in call_args


@pytest.mark.asyncio
async def test_execute_env_vars(backend, profile):
    handle = DockerTrialHandle(trial_id="t1", workspace=Path("/ws"), container_id="cid")
    exec_proc = _mock_process()

    with patch("asyncio.create_subprocess_exec", return_value=exec_proc) as mock:
        await backend.execute(handle, profile, ["echo"], timeout=5, env={"FOO": "bar"})

    call_args = mock.call_args[0]
    assert "-e" in call_args
    idx = list(call_args).index("-e")
    assert call_args[idx + 1] == "FOO=bar"


@pytest.mark.asyncio
async def test_execute_wrong_handle_type(backend, profile):
    handle = TrialHandle(trial_id="t1", workspace=Path("/ws"))
    with pytest.raises(TypeError):
        await backend.execute(handle, profile, ["echo"], timeout=5)


@pytest.mark.asyncio
async def test_destroy(backend):
    handle = DockerTrialHandle(trial_id="t1", workspace=Path("/ws"), container_id="cid")
    rm_proc = _mock_process()

    with patch("asyncio.create_subprocess_exec", return_value=rm_proc) as mock:
        await backend.destroy(handle)

    call_args = mock.call_args[0]
    assert "rm" in call_args
    assert "-f" in call_args
    assert "cid" in call_args


@pytest.mark.asyncio
async def test_destroy_plain_handle_is_noop(backend):
    handle = TrialHandle(trial_id="t1", workspace=Path("/ws"))
    await backend.destroy(handle)  # should not raise
