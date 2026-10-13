from pathlib import Path
from typing import Any

from sandbox_server.isolation.base import IsolationBackend, TrialHandle
from sandbox_server.profiles import SandboxProfile
from sandbox_server.workspace import WorkspaceManager

ANSWER_SENTINEL = "__RETURN_ANSWER__"


class ActionDispatcher:
    """Routes tool calls: file tools → workspace, execution tools
    (bash/python) → isolation backend."""

    def __init__(self, workspace_manager: WorkspaceManager):
        self.workspace_manager = workspace_manager

    async def execute(
        self,
        backend: IsolationBackend,
        profile: SandboxProfile,
        handle: TrialHandle,
        tool: str,
        args: dict[str, Any],
        timeout: float,
    ) -> Any:
        if tool not in profile.enabled_tools:
            raise ValueError(f"tool {tool!r} is not enabled for profile {profile.name!r}")
        if tool == "return_answer":
            return {"answer": str(args.get("answer", "")), "_sentinel": ANSWER_SENTINEL}
        if tool == "read_file":
            return self._read_file(handle.workspace, args)
        if tool == "write_file":
            return self._write_file(handle.workspace, args)
        if tool == "str_replace_file":
            return self._replace_file(handle.workspace, args)
        if tool == "bash":
            return await self._run(backend, profile, handle, ["/bin/bash", "-lc", args["command"]], timeout)
        if tool == "execute_python":
            return await self._execute_python(backend, profile, handle, args, timeout)
        raise ValueError(f"no dispatcher for tool {tool!r}")

    def _read_file(self, workspace: Path, args: dict[str, Any]) -> str:
        path = self.workspace_manager.resolve(workspace, args["path"])
        if not path.is_file():
            raise FileNotFoundError(f"file not found: {args['path']}")
        return path.read_text(encoding="utf-8")

    def _write_file(self, workspace: Path, args: dict[str, Any]) -> dict[str, Any]:
        path = self.workspace_manager.resolve(workspace, args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8")
        return {"ok": True, "path": str(path.relative_to(workspace))}

    def _replace_file(self, workspace: Path, args: dict[str, Any]) -> dict[str, Any]:
        path = self.workspace_manager.resolve(workspace, args["path"])
        if not path.is_file():
            raise FileNotFoundError(f"file not found: {args['path']}")
        content = path.read_text(encoding="utf-8")
        old = args["old_str"]
        if old not in content:
            raise ValueError(f"old_str not found in {args['path']}")
        path.write_text(content.replace(old, args["new_str"], 1), encoding="utf-8")
        return {"ok": True, "path": str(path.relative_to(workspace))}

    async def _execute_python(
        self,
        backend: IsolationBackend,
        profile: SandboxProfile,
        handle: TrialHandle,
        args: dict[str, Any],
        timeout: float,
    ) -> str:
        # Write to a known path in the trial workspace; every backend exposes
        # the workspace at /workspace with that directory as the cwd.
        script = self.workspace_manager.resolve(handle.workspace, ".sandbox/action.py")
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(str(args["code"]), encoding="utf-8")
        return await self._run(backend, profile, handle, [profile.python_executable, ".sandbox/action.py"], timeout)

    async def _run(
        self,
        backend: IsolationBackend,
        profile: SandboxProfile,
        handle: TrialHandle,
        argv: list[str],
        timeout: float,
    ) -> str:
        result = await backend.execute(handle, profile, argv, timeout)
        if result.returncode != 0:
            detail = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
            raise RuntimeError(f"command exited {result.returncode}: {detail}")
        if result.stderr.strip():
            return f"{result.stdout}\n[stderr]\n{result.stderr}" if result.stdout else result.stderr
        return result.stdout
