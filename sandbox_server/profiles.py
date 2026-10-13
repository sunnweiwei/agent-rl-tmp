from dataclasses import dataclass, field
from pathlib import Path

from sandbox_server.schemas import DEFAULT_REWARD_TIMEOUT_SECONDS

DEFAULT_TOOLS = frozenset({"bash", "execute_python", "read_file", "write_file", "str_replace_file", "return_answer"})


@dataclass(frozen=True)
class SandboxProfile:
    """Determines which backend, tools, and resource limits apply to trials
    created under this profile."""

    name: str
    backend_type: str = "nsjail"
    version: str = "1"
    enabled_tools: frozenset[str] = DEFAULT_TOOLS
    max_live_trials: int = 1024
    max_running_actions: int = 64
    max_queued_actions: int | None = None
    tool_max_running_actions: dict[str, int] = field(default_factory=dict)
    lease_seconds: float = 900.0
    rootfs: Path | None = None
    nsjail_config: Path | None = None
    task_root: Path | None = None
    python_executable: str = "python3"
    trusted_subprocess: bool = False
    setup_timeout_seconds: float = 300.0
    reward_timeout_seconds: float = DEFAULT_REWARD_TIMEOUT_SECONDS


def default_profiles(cfg) -> dict[str, SandboxProfile]:
    """One profile per isolation backend. Both share the task root and limits."""
    shared = dict(
        max_live_trials=cfg.max_live_trials,
        max_running_actions=cfg.max_running_actions,
        max_queued_actions=cfg.max_queued_actions,
        tool_max_running_actions=cfg.tool_max_running_actions,
        lease_seconds=cfg.lease_seconds,
        task_root=cfg.task_root,
        reward_timeout_seconds=cfg.reward_timeout_seconds,
    )
    return {
        "nsjail": SandboxProfile(
            name="nsjail",
            backend_type="nsjail",
            rootfs=cfg.profile_rootfs,
            nsjail_config=cfg.nsjail_config,
            python_executable="/usr/local/bin/python3",
            **shared,
        ),
        "docker": SandboxProfile(name="docker", backend_type="docker", **shared),
    }
