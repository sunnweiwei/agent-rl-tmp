import shutil
from pathlib import Path


class WorkspaceError(ValueError):
    pass


class WorkspaceManager:
    """Creates and destroys per-trial workspace directories.
    All file path resolution is confined to the trial's workspace."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, session_id: str, trial_id: str) -> Path:
        path = self.root / session_id / trial_id
        path.mkdir(parents=True, exist_ok=False)
        return path

    def resolve(self, workspace: Path, requested: str) -> Path:
        """Resolve a relative path within the workspace. Rejects absolute
        paths, traversal (../), and symlinks."""
        relative = Path(requested)
        if relative.is_absolute():
            raise WorkspaceError("absolute paths are not allowed")
        candidate = (workspace / relative).resolve(strict=False)
        try:
            candidate.relative_to(workspace.resolve())
        except ValueError as exc:
            raise WorkspaceError("path escapes trial workspace") from exc

        current = workspace.resolve()
        for part in relative.parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise WorkspaceError("symlink traversal is not allowed")
        if candidate.is_symlink():
            raise WorkspaceError("symlink targets are not allowed")
        return candidate

    def destroy(self, workspace: Path) -> None:
        try:
            resolved = workspace.resolve()
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError("refusing to remove path outside workspace root") from exc
        shutil.rmtree(resolved, ignore_errors=True)

        session_dir = resolved.parent
        try:
            session_dir.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError("refusing to remove path outside workspace root") from exc
        if session_dir != self.root:
            try:
                session_dir.rmdir()
            except OSError:
                pass
