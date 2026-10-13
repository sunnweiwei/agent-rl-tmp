from collections import defaultdict
import threading
import time

from verl.trainer.main_ppo_sync import ReplayBuffer


def _make_replay_buffer() -> ReplayBuffer:
    replay_buffer = object.__new__(ReplayBuffer)
    replay_buffer.partitions = defaultdict(dict)
    replay_buffer.poll_interval = 0.0
    replay_buffer.lock = threading.Lock()
    return replay_buffer


def test_sample_ready_prompt_groups_keeps_incomplete_current_groups_pending():
    replay_buffer = _make_replay_buffer()
    replay_buffer.add(
        "train",
        {
            "p0": {"status": "finished", "global_steps": 1, "expected_rollout_n": 2},
            "p0_0_0": {"status": "success", "global_steps": 1, "seq_len": 10},
            "p0_1_0": {"status": "success", "global_steps": 1, "seq_len": 11},
            "p1": {"status": "running", "global_steps": 1, "expected_rollout_n": 2},
            "p1_0_0": {"status": "success", "global_steps": 1, "seq_len": 12},
        },
    )

    batch = replay_buffer.sample_ready_prompt_groups(
        partition_id="train",
        current_prompt_keys=["p0", "p1"],
        current_global_steps=1,
        min_current_prompt_groups=1,
        max_off_policy_steps=5,
        default_rollout_n=2,
        ready_grace_wait_seconds=0,
    )

    assert batch.keys == ["p0_0_0", "p0_1_0"]
    assert batch.extra_info["grouped_partial/current_selected_prompt_groups"] == 1
    assert batch.extra_info["grouped_partial/selected_output_rows"] == 2
    assert batch.extra_info["grouped_partial/selected_prompt_groups_missing_rollouts"] == 0


def test_historical_ready_groups_do_not_satisfy_current_minimum():
    replay_buffer = _make_replay_buffer()
    replay_buffer.poll_interval = 0.001
    replay_buffer.add(
        "train",
        {
            "old": {"status": "finished", "global_steps": 0, "expected_rollout_n": 2},
            "old_0_0": {"status": "success", "global_steps": 0, "seq_len": 10},
            "old_1_0": {"status": "success", "global_steps": 0, "seq_len": 11},
            "new": {"status": "running", "global_steps": 1, "expected_rollout_n": 2},
        },
    )
    result = {}

    def sample():
        result["batch"] = replay_buffer.sample_ready_prompt_groups(
            partition_id="train",
            current_prompt_keys=["new"],
            current_global_steps=1,
            min_current_prompt_groups=1,
            max_off_policy_steps=5,
            default_rollout_n=2,
            ready_grace_wait_seconds=0,
        )

    thread = threading.Thread(target=sample)
    thread.start()
    time.sleep(0.02)
    assert thread.is_alive()

    replay_buffer.add(
        "train",
        {
            "new": {"status": "finished", "global_steps": 1, "expected_rollout_n": 2},
            "new_0_0": {"status": "success", "global_steps": 1, "seq_len": 12},
            "new_1_0": {"status": "success", "global_steps": 1, "seq_len": 13},
        },
    )
    thread.join(timeout=1)
    assert not thread.is_alive()

    batch = result["batch"]
    assert batch.keys == ["old_0_0", "old_1_0", "new_0_0", "new_1_0"]
    assert batch.extra_info["grouped_partial/historical_selected_prompt_groups"] == 1
    assert batch.extra_info["grouped_partial/current_selected_prompt_groups"] == 1


def test_stale_historical_group_blocks_until_ready():
    replay_buffer = _make_replay_buffer()
    replay_buffer.poll_interval = 0.001
    replay_buffer.add(
        "train",
        {
            "stale": {"status": "finished", "global_steps": 1, "expected_rollout_n": 2},
            "stale_0_0": {"status": "success", "global_steps": 1, "seq_len": 10},
            "new": {"status": "finished", "global_steps": 6, "expected_rollout_n": 2},
            "new_0_0": {"status": "success", "global_steps": 6, "seq_len": 12},
            "new_1_0": {"status": "success", "global_steps": 6, "seq_len": 13},
        },
    )
    result = {}

    def sample():
        result["batch"] = replay_buffer.sample_ready_prompt_groups(
            partition_id="train",
            current_prompt_keys=["new"],
            current_global_steps=6,
            min_current_prompt_groups=1,
            max_off_policy_steps=5,
            default_rollout_n=2,
            ready_grace_wait_seconds=0,
        )

    thread = threading.Thread(target=sample)
    thread.start()
    time.sleep(0.02)
    assert thread.is_alive()

    replay_buffer.add(
        "train",
        {
            "stale_1_0": {"status": "success", "global_steps": 1, "seq_len": 11},
        },
    )
    thread.join(timeout=1)
    assert not thread.is_alive()

    batch = result["batch"]
    assert batch.keys == ["stale_0_0", "stale_1_0", "new_0_0", "new_1_0"]
    assert batch.extra_info["grouped_partial/stale_wait_polls"] > 0
