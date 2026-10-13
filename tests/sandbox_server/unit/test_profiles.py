import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from sandbox_server.profiles import DEFAULT_TOOLS, SandboxProfile, default_profiles


def _cfg(**overrides):
    values = dict(
        max_live_trials=10,
        max_running_actions=4,
        max_queued_actions=100,
        tool_max_running_actions={"bash": 2},
        lease_seconds=60.0,
        task_root=Path("/tasks"),
        reward_timeout_seconds=30.0,
        profile_rootfs=Path("/rootfs"),
        nsjail_config=Path("/nsjail.cfg"),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_default_profiles_one_per_backend():
    profiles = default_profiles(_cfg())
    assert {name: p.backend_type for name, p in profiles.items()} == {"nsjail": "nsjail", "docker": "docker"}


def test_default_profiles_share_limits_and_task_root():
    for profile in default_profiles(_cfg()).values():
        assert profile.enabled_tools == DEFAULT_TOOLS
        assert profile.max_live_trials == 10
        assert profile.max_running_actions == 4
        assert profile.tool_max_running_actions == {"bash": 2}
        assert profile.task_root == Path("/tasks")
        assert profile.reward_timeout_seconds == 30.0


def test_nsjail_profile_uses_rootfs():
    profile = default_profiles(_cfg())["nsjail"]
    assert profile.rootfs == Path("/rootfs")
    assert profile.nsjail_config == Path("/nsjail.cfg")


def test_profile_is_frozen():
    profile = SandboxProfile(name="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.name = "y"
