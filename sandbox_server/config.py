import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

_CONFIG_PATH = Path(os.getenv("SANDBOX_CONFIG", "/etc/sandbox/config.yaml"))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value) if value else None


def _parse_bytes(value: str | int | None, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    multipliers = {
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
    }
    for suffix, multiplier in sorted(multipliers.items(), key=lambda item: -len(item[0])):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)].strip()) * multiplier)
    return int(float(text))


def _env_bytes(name: str, default: str | int) -> int:
    return _parse_bytes(os.getenv(name), _parse_bytes(default, 0))


def _default_thread_pool_workers() -> int:
    cpu_count = os.cpu_count() or 4
    return min(1024, max(32, cpu_count * 4))


def _env_int_map(name: str) -> dict[str, int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    result: dict[str, int] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(f"{name} entries must be tool:limit pairs")
        key, value = item.split(":", 1)
        result[key.strip()] = int(value.strip())
    return result


class SandboxConfig(BaseModel):
    """Server configuration from env vars and optional YAML override."""

    # Event-loop default executor used for subprocess waits and reward
    # functions that run in worker threads.
    thread_pool_workers: int = Field(
        default_factory=lambda: _env_int("SANDBOX_THREAD_POOL_WORKERS", _default_thread_pool_workers())
    )
    allow_unsafe_subprocess: bool = Field(
        default_factory=lambda: os.getenv("SANDBOX_ALLOW_UNSAFE_SUBPROCESS", "0") == "1"
    )
    enabled_profiles: list[str] | None = Field(
        default_factory=lambda: [
            item.strip() for item in os.getenv("SANDBOX_ENABLED_PROFILES", "").split(",") if item.strip()
        ]
        or None
    )

    # Tasks and workspaces
    task_root: Path | None = Field(default_factory=lambda: _env_path("SANDBOX_TASK_ROOT"))
    workspace_root: Path = Field(
        default_factory=lambda: Path(os.getenv("SANDBOX_WORKSPACE_ROOT", "/tmp/agent-sandbox"))
    )

    # Capacity, shared by all profiles
    max_live_trials: int = Field(default_factory=lambda: _env_int("SANDBOX_MAX_LIVE_TRIALS", 1024))
    max_running_actions: int = Field(default_factory=lambda: _env_int("SANDBOX_MAX_RUNNING_ACTIONS", 128))
    max_queued_actions: int = Field(default_factory=lambda: _env_int("SANDBOX_MAX_QUEUED_ACTIONS", 50000))
    tool_max_running_actions: dict[str, int] = Field(
        default_factory=lambda: _env_int_map("SANDBOX_TOOL_MAX_RUNNING_ACTIONS")
    )
    lease_seconds: float = Field(default_factory=lambda: _env_float("SANDBOX_LEASE_SECONDS", 900.0))
    reward_timeout_seconds: float = Field(default_factory=lambda: _env_float("SANDBOX_REWARD_TIMEOUT_SECONDS", 600.0))

    # nsjail backend
    nsjail_executable: str = Field(default_factory=lambda: os.getenv("NSJAIL_EXECUTABLE", "nsjail"))
    profile_rootfs: Path | None = Field(default_factory=lambda: _env_path("SANDBOX_PROFILE_ROOTFS"))
    nsjail_config: Path | None = Field(default_factory=lambda: _env_path("SANDBOX_NSJAIL_CONFIG"))

    # Docker backend
    docker_executable: str = Field(default_factory=lambda: os.getenv("DOCKER_EXECUTABLE", "docker"))
    docker_default_image: str = Field(default_factory=lambda: os.getenv("SANDBOX_DOCKER_IMAGE", "python:3.12-slim"))
    docker_mem_limit: str = Field(default_factory=lambda: os.getenv("SANDBOX_DOCKER_MEM_LIMIT", "8g"))
    docker_cpu_limit: float = Field(default_factory=lambda: _env_float("SANDBOX_DOCKER_CPU_LIMIT", 2.0))
    docker_pids_limit: int = Field(default_factory=lambda: _env_int("SANDBOX_DOCKER_PIDS_LIMIT", 512))
    # "none" = fully isolated (safe default).
    docker_network: str = Field(default_factory=lambda: os.getenv("SANDBOX_DOCKER_NETWORK", "none"))

    # Process/container memory guard. Soft limit rejects new work before Docker
    # or Ray OOM killers get involved. Trim threshold asks glibc to return free
    # arenas after large bursts; it is best-effort.
    sandbox_memory_soft_limit_bytes: int = Field(
        default_factory=lambda: _env_bytes("SANDBOX_MEMORY_SOFT_LIMIT", "192g")
    )
    sandbox_memory_trim_threshold_bytes: int = Field(
        default_factory=lambda: _env_bytes("SANDBOX_MEMORY_TRIM_THRESHOLD", "64g")
    )

    lease_check_interval: float = 5.0


_config: SandboxConfig | None = None


def get_config() -> SandboxConfig:
    global _config
    if _config is None:
        overrides = {}
        if _CONFIG_PATH.exists():
            overrides = yaml.safe_load(_CONFIG_PATH.read_text()) or {}
        _config = SandboxConfig(**overrides)
    return _config
