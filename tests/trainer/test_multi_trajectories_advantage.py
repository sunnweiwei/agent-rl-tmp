# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import defaultdict

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import verl.trainer.main_ppo_sync as main_ppo_sync
from verl.protocol import DataProto
from verl.trainer.main_ppo_sync import PPOTrainer, compute_advantage, compute_advantage_for_multi_trajectories
from verl.trainer.ppo.core_algos import AdvantageEstimator


@pytest.fixture
def batch_data() -> DataProto:
    tensors = {
        "token_level_rewards": torch.tensor(
            [
                [100, 0, 0, 0],
                [1, 2, 3, 4],
                [200, 200, 0, 0],
                [150, 0, 0, 150],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ],
            dtype=torch.float32,
        ),
        "response_mask": torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 1, 0, 1],
                [0, 0, 0, 0],
            ],
            dtype=torch.long,
        ),
    }
    non_tensors = {
        "uid": np.array(["prompt_a"] * 6, dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def test_compute_advantage_for_single_trajectory(batch_data: DataProto):
    result = compute_advantage_for_multi_trajectories(
        data=batch_data,
        batch_keys=[f"prompt_a_{i}_0" for i in range(len(batch_data))],
        adv_estimator=AdvantageEstimator.GRPO,
    )
    expected = compute_advantage(
        batch_data,
        adv_estimator=AdvantageEstimator.GRPO,
    )
    assert torch.equal(result.batch["advantages"], expected.batch["advantages"])
    assert torch.equal(result.batch["returns"], expected.batch["returns"])


def test_compute_advantage_for_multi_trajectories(batch_data: DataProto):
    result = compute_advantage_for_multi_trajectories(
        data=batch_data,
        batch_keys=["prompt_a_0_0", "prompt_a_0_1", "prompt_a_2_0", "prompt_a_2_1", "prompt_a_3_0", "prompt_a_4_0"],
        adv_estimator=AdvantageEstimator.GRPO,
    )
    expected = compute_advantage(
        batch_data.select_idxs([1, 3, 4, 5]),
        adv_estimator=AdvantageEstimator.GRPO,
    )
    gather_row_indices = [0, 0, 1, 1, 2, 3]
    gather_col_indices = [0, 0, 1, 1, 0, 0]
    adv_expected = (
        expected.batch["advantages"][gather_row_indices, gather_col_indices].unsqueeze(-1)
        * result.batch["response_mask"]
    )
    assert torch.equal(result.batch["advantages"], adv_expected)
    assert torch.equal(result.batch["returns"], adv_expected)


def test_compute_advantage_adds_action_local_process_advantages():
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            "response_mask": torch.ones((2, 2), dtype=torch.long),
            "process_advantages": torch.tensor([[0.0, -0.25], [0.0, 0.0]], dtype=torch.float32),
        },
        non_tensors={"uid": np.array(["prompt-a", "prompt-a"], dtype=object)},
    )

    result = compute_advantage_for_multi_trajectories(
        data=data,
        batch_keys=["prompt-a_0_0", "prompt-a_1_0"],
        adv_estimator=AdvantageEstimator.GRPO,
        norm_adv_by_std_in_grpo=False,
    )

    expected = torch.tensor([[-0.5, -0.75], [0.5, 0.5]], dtype=torch.float32)
    torch.testing.assert_close(result.batch["advantages"], expected)
    torch.testing.assert_close(result.batch["returns"], expected)


def test_curriculum_weighting_preserves_sign_and_downweights_saturated_tasks():
    trainer = object.__new__(PPOTrainer)
    trainer._curriculum_success_ema = {}
    trainer._curriculum_observations = defaultdict(int)
    config = {
        "ema_decay": 0.75,
        "min_observations": 2,
        "lower_success_rate": 0.05,
        "upper_success_rate": 0.85,
        "inactive_weight": 0.25,
    }
    keys = ["uid-a_0_0", "uid-a_1_0", "uid-b_0_0", "uid-b_1_0"]
    task_ids = ["task-a", "task-a", "task-b", "task-b"]
    static_weights = [2.0, 2.0, 0.5, 0.5]

    def make_data():
        return DataProto.from_dict(
            tensors={
                "advantages": torch.tensor([[1.0], [-1.0], [1.0], [-1.0]]),
                "returns": torch.tensor([[1.0], [-1.0], [1.0], [-1.0]]),
                "token_level_scores": torch.tensor([[0.0], [0.0], [1.0], [1.0]]),
            }
        )

    first = make_data()
    trainer._apply_curriculum_advantage_weights(
        data=first,
        batch_keys=keys,
        task_ids=task_ids,
        static_weights=static_weights,
        base_reward_scores=[0.0, 0.0, 1.0, 1.0],
        config=config,
        metrics={},
    )
    torch.testing.assert_close(
        first.batch["advantages"], torch.tensor([[2.0], [-2.0], [0.5], [-0.5]])
    )

    metrics = {}
    second = make_data()
    trainer._apply_curriculum_advantage_weights(
        data=second,
        batch_keys=keys,
        task_ids=task_ids,
        static_weights=static_weights,
        base_reward_scores=[0.0, 0.0, 1.0, 1.0],
        config=config,
        metrics=metrics,
    )
    torch.testing.assert_close(
        second.batch["advantages"], torch.tensor([[0.5], [-0.5], [0.125], [-0.125]])
    )
    assert metrics["curriculum/inactive_task_fraction"] == 1.0


def test_bypass_old_log_prob_preserves_process_advantage_field(monkeypatch):
    trainer = object.__new__(PPOTrainer)
    trainer.config = type(
        "Config",
        (),
        {"algorithm": {"rollout_correction": {"bypass_mode": True}}},
    )()
    batch = type(
        "Batch",
        (),
        {
            "keys": ["sample-0"],
            "partition_id": "train",
            "fields": None,
            "extra_info": {"temperature": 1.0},
        },
    )()

    rollout_log_probs = torch.tensor([[-0.1, -0.2]], dtype=torch.float32)
    stored = TensorDict(
        {"rollout_log_probs": rollout_log_probs.clone()},
        batch_size=1,
    )

    def fake_get(*, keys, partition_id, select_fields):
        assert keys == ["sample-0"]
        assert partition_id == "train"
        assert select_fields == ["rollout_log_probs"]
        return stored.clone()

    def fake_put(*, keys, partition_id, fields):
        assert keys == ["sample-0"]
        assert partition_id == "train"
        torch.testing.assert_close(fields["old_log_probs"], rollout_log_probs)
        return type(
            "UpdatedBatch",
            (),
            {
                "keys": keys,
                "partition_id": partition_id,
                "fields": ["old_log_probs", "process_advantages", "response_mask"],
                "extra_info": {},
            },
        )()

    monkeypatch.setattr(main_ppo_sync.tq, "kv_batch_get", fake_get)
    monkeypatch.setattr(main_ppo_sync.tq, "kv_batch_put", fake_put)

    result = trainer._compute_old_log_prob(batch, metrics={})

    assert "process_advantages" in result.fields
    assert result.extra_info == {"temperature": 1.0}


def test_compute_advantage_for_foldgrpo_multi_outputs():
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.tensor([[1.0], [1.0], [3.0], [5.0]], dtype=torch.float32),
            "response_mask": torch.ones((4, 1), dtype=torch.long),
        },
        non_tensors={
            "uid": np.array(["prompt-a"] * 4, dtype=object),
            "gen_uid": np.array(["rollout-0", "rollout-0", "rollout-1", "rollout-2"], dtype=object),
        },
    )

    result = compute_advantage_for_multi_trajectories(
        data=data,
        batch_keys=["prompt-a_0_0", "prompt-a_0_1", "prompt-a_1_0", "prompt-a_2_0"],
        adv_estimator=AdvantageEstimator.FOLDGRPO,
        norm_adv_by_std_in_grpo=False,
    )

    expected = torch.tensor([[-2.0], [-2.0], [0.0], [2.0]], dtype=torch.float32)
    assert torch.equal(result.batch["advantages"], expected)
    assert torch.equal(result.batch["returns"], expected)


def test_val_metrics_update_logs_all_metrics_globally():
    trainer = object.__new__(PPOTrainer)

    metrics = trainer._val_metrics_update(
        data_sources=["source-a", "source-b"],
        sample_uids=["uid-a", "uid-b"],
        reward_extra_infos_dict={
            "reward": [1.0, 3.0],
            "acc": [1.0, 0.0],
            "all/score": [2.0, 4.0],
        },
        sample_turns=[],
    )

    assert metrics["val-core/all/score"] == 3.0
    assert not any(key.startswith("val-aux/source-a/all/score") for key in metrics)
