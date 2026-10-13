import pytest

from sandbox_server.actions import ANSWER_SENTINEL, ActionDispatcher
from sandbox_server.isolation.base import ExecutionResult, TrialHandle
from sandbox_server.profiles import DEFAULT_TOOLS, SandboxProfile
from sandbox_server.workspace import WorkspaceManager
from tests.sandbox_server.conftest import MockBackend


@pytest.fixture
def ws_manager(tmp_path):
    return WorkspaceManager(tmp_path / "ws")


@pytest.fixture
def workspace(ws_manager):
    return ws_manager.create("s", "t")


@pytest.fixture
def handle(workspace):
    return TrialHandle(trial_id="t", workspace=workspace)


@pytest.fixture
def profile():
    return SandboxProfile(name="test", backend_type="mock", enabled_tools=DEFAULT_TOOLS)


@pytest.fixture
def dispatcher(ws_manager):
    return ActionDispatcher(ws_manager)


@pytest.mark.asyncio
async def test_rejects_disabled_tool(dispatcher, handle):
    profile = SandboxProfile(name="t", enabled_tools=frozenset({"bash"}))
    with pytest.raises(ValueError, match="not enabled"):
        await dispatcher.execute(MockBackend(), profile, handle, "execute_python", {"code": "1"}, 10)


@pytest.mark.asyncio
async def test_return_answer(dispatcher, handle, profile):
    result = await dispatcher.execute(MockBackend(), profile, handle, "return_answer", {"answer": "42"}, 10)
    assert result == {"answer": "42", "_sentinel": ANSWER_SENTINEL}


@pytest.mark.asyncio
async def test_read_file(dispatcher, handle, profile, workspace):
    (workspace / "a.txt").write_text("hello")
    result = await dispatcher.execute(MockBackend(), profile, handle, "read_file", {"path": "a.txt"}, 10)
    assert result == "hello"


@pytest.mark.asyncio
async def test_read_file_not_found(dispatcher, handle, profile):
    with pytest.raises(FileNotFoundError):
        await dispatcher.execute(MockBackend(), profile, handle, "read_file", {"path": "missing.txt"}, 10)


@pytest.mark.asyncio
async def test_write_file_creates_parents(dispatcher, handle, profile, workspace):
    result = await dispatcher.execute(
        MockBackend(), profile, handle, "write_file", {"path": "a/b/c.txt", "content": "x"}, 10
    )
    assert result == {"ok": True, "path": "a/b/c.txt"}
    assert (workspace / "a/b/c.txt").read_text() == "x"


@pytest.mark.asyncio
async def test_write_file_rejects_escape(dispatcher, handle, profile):
    with pytest.raises(ValueError, match="escapes"):
        await dispatcher.execute(MockBackend(), profile, handle, "write_file", {"path": "../x", "content": "x"}, 10)


@pytest.mark.asyncio
async def test_str_replace_file(dispatcher, handle, profile, workspace):
    (workspace / "f.py").write_text("a = 1\na = 1\n")
    await dispatcher.execute(
        MockBackend(), profile, handle, "str_replace_file", {"path": "f.py", "old_str": "a = 1", "new_str": "a = 2"}, 10
    )
    assert (workspace / "f.py").read_text() == "a = 2\na = 1\n"


@pytest.mark.asyncio
async def test_str_replace_missing_old(dispatcher, handle, profile, workspace):
    (workspace / "f.py").write_text("x")
    with pytest.raises(ValueError, match="old_str not found"):
        await dispatcher.execute(
            MockBackend(), profile, handle, "str_replace_file", {"path": "f.py", "old_str": "y", "new_str": "z"}, 10
        )


@pytest.mark.asyncio
async def test_bash_delegates_to_backend(dispatcher, handle, profile):
    backend = MockBackend(ExecutionResult(0, "out\n", ""))
    result = await dispatcher.execute(backend, profile, handle, "bash", {"command": "echo out"}, 10)
    assert result == "out\n"
    assert backend.executed == [("t", ["/bin/bash", "-lc", "echo out"])]


@pytest.mark.asyncio
async def test_bash_includes_stderr(dispatcher, handle, profile):
    backend = MockBackend(ExecutionResult(0, "out", "warn"))
    result = await dispatcher.execute(backend, profile, handle, "bash", {"command": "x"}, 10)
    assert result == "out\n[stderr]\nwarn"


@pytest.mark.asyncio
async def test_nonzero_exit_raises_with_output(dispatcher, handle, profile):
    backend = MockBackend(ExecutionResult(2, "partial", "Traceback: boom"))
    with pytest.raises(RuntimeError, match="command exited 2: partial\nTraceback: boom"):
        await dispatcher.execute(backend, profile, handle, "bash", {"command": "x"}, 10)


@pytest.mark.asyncio
async def test_execute_python_writes_script(dispatcher, handle, profile, workspace):
    backend = MockBackend(ExecutionResult(0, "3\n", ""))
    result = await dispatcher.execute(backend, profile, handle, "execute_python", {"code": "print(1 + 2)"}, 10)
    assert result == "3\n"
    assert (workspace / ".sandbox/action.py").read_text() == "print(1 + 2)"
    assert backend.executed == [("t", ["python3", ".sandbox/action.py"])]


@pytest.mark.asyncio
async def test_unknown_tool(dispatcher, handle):
    profile = SandboxProfile(name="t", enabled_tools=frozenset({"nope"}))
    with pytest.raises(ValueError, match="no dispatcher"):
        await dispatcher.execute(MockBackend(), profile, handle, "nope", {}, 10)
