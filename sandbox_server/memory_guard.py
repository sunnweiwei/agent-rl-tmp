from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemorySnapshot:
    process_rss_bytes: int
    cgroup_current_bytes: int | None
    cgroup_max_bytes: int | None

    @property
    def current_bytes(self) -> int:
        return max(
            self.process_rss_bytes,
            self.cgroup_current_bytes or 0,
        )


def snapshot() -> MemorySnapshot:
    return MemorySnapshot(
        process_rss_bytes=_process_rss_bytes(),
        cgroup_current_bytes=_read_cgroup_int("memory.current"),
        cgroup_max_bytes=_read_cgroup_int("memory.max"),
    )


def over_soft_limit(soft_limit_bytes: int) -> tuple[bool, MemorySnapshot]:
    snap = snapshot()
    if soft_limit_bytes <= 0:
        return False, snap
    return snap.current_bytes >= soft_limit_bytes, snap


def pressure_error(soft_limit_bytes: int, snap: MemorySnapshot) -> str:
    return (
        "MemoryPressure: sandbox memory soft limit exceeded "
        f"(current={_format_bytes(snap.current_bytes)}, "
        f"process_rss={_format_bytes(snap.process_rss_bytes)}, "
        f"cgroup_current={_format_optional_bytes(snap.cgroup_current_bytes)}, "
        f"cgroup_max={_format_optional_bytes(snap.cgroup_max_bytes)}, "
        f"soft_limit={_format_bytes(soft_limit_bytes)})"
    )


def maybe_trim(trim_threshold_bytes: int) -> None:
    if trim_threshold_bytes <= 0:
        return
    snap = snapshot()
    if snap.process_rss_bytes < trim_threshold_bytes:
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        return


def _process_rss_bytes() -> int:
    try:
        with Path("/proc/self/status").open() as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return 0
    return 0


def _read_cgroup_int(filename: str) -> int | None:
    for base in (Path("/sys/fs/cgroup"),):
        path = base / filename
        try:
            text = path.read_text().strip()
        except OSError:
            continue
        if not text or text == "max":
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _format_optional_bytes(value: int | None) -> str:
    return "unknown" if value is None else _format_bytes(value)


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if abs(size) < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{value}B"
