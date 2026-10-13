# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Synchronous PPO trainer with colocated actor and rollout.

Differs from original PPO trainer in main_ppo.py:
1. Use TransferQueue to zero-padding and zero-copy data transfer.
2. Use ReplayBuffer to sample data from TransferQueue.
3. Support different `n` sampling for each prompt.
4. Support multiple outputs for each agent loop.
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pprint import pprint
from typing import Any

import hydra
import numpy as np
import ray
import torch

try:
    import transfer_queue as tq
    from transfer_queue import KVBatchMeta
except ImportError:
    print("Please install TQ by calling `pip install TransferQueue==0.1.8` and try again.")
    from verl.utils.transferqueue_utils import KVBatchMeta, tq

from omegaconf import DictConfig, OmegaConf, open_dict
from tensordict import NonTensorData, NonTensorStack, TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    get_trajectory_info,
)
from verl.experimental.reward_loop import RewardLoopManager
from verl.experimental.teacher_loop import MultiTeacherModelManager
from verl.protocol import DataProto, DataProtoFuture
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayWorkerGroup,
    ResourcePoolManager,
    create_colocated_worker_cls,
)
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler, run_ppo
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.padding_utils import upsample_batch_to_divisible_size
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_advantage, compute_spec_decode_metrics
from verl.trainer.ppo.rollout_corr_helper import (
    compute_compact_offpolicy_metrics,
    compute_detailed_offpolicy_metrics,
    compute_rollout_correction_and_add_to_batch,
)
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_teacher_policy
from verl.utils import hf_processor, hf_tokenizer
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.debug import marked_timer
from verl.utils.debug.metrics import calculate_debug_metrics
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.model import compute_position_id_with_mask
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.ray_utils import auto_await
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.tensordict_utils import list_of_dict_to_tensordict
from verl.utils.tracking import Tracking, ValidationGenerationsLogger
from verl.workers.config import CriticConfig, DistillationConfig
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker, TrainingWorkerConfig
from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager
from verl.workers.rollout.replica import TokenOutput
from verl.workers.utils.losses import value_loss
from verl.workers.utils.padding import response_from_nested, response_to_nested


def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
    params["top_p"] = 1.0
    params["top_k"] = -1
    params["temperature"] = 0


def to_python_list(value: Any) -> list:
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def int_or_default(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def token_ids_digest(token_ids: Any) -> str:
    """Return a stable digest for an exact token-id sequence."""
    if isinstance(token_ids, torch.Tensor):
        tensor = token_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
        payload = tensor.numpy().tobytes()
    else:
        payload = np.asarray([int_or_default(item, 0) for item in to_python_list(token_ids)], dtype=np.int64).tobytes()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# ======================================= USER SECTION BEGIN =======================================


GROUPED_PARTIAL_METRIC_ALIASES = {
    "grouped_partial/current_selected_prompt_groups": "partial_rollout/current_selected_prompt_groups",
    "grouped_partial/historical_selected_prompt_groups": "partial_rollout/historical_selected_prompt_groups",
    "grouped_partial/selected_prompt_groups": "partial_rollout/selected_prompt_groups",
    "grouped_partial/selected_output_rows": "partial_rollout/selected_output_rows",
    "grouped_partial/ready_grace_waited_seconds": "partial_rollout/ready_grace_waited_seconds",
    "grouped_partial/current_ready_prompt_groups_added_by_grace": (
        "partial_rollout/current_ready_prompt_groups_added_by_grace"
    ),
    "grouped_partial/stale_wait_polls": "partial_rollout/stale_wait_polls",
    "grouped_partial/selected_prompt_groups_missing_rollouts": (
        "partial_rollout/selected_prompt_groups_missing_rollouts"
    ),
    "grouped_partial/selected_prompt_groups_extra_rollouts": "partial_rollout/selected_prompt_groups_extra_rollouts",
}


@ray.remote(max_concurrency=1000)
class RolloutGate:
    """Shared gate that blocks generation calls while colocated GPUs are used for training."""

    def __init__(self, opened: bool = True, poll_interval: float = 0.25):
        self.opened = opened
        self.poll_interval = poll_interval

    async def open(self) -> None:
        self.opened = True

    async def close(self) -> None:
        self.opened = False

    async def is_open(self) -> bool:
        return self.opened

    async def wait_open(self) -> None:
        while not self.opened:
            await asyncio.sleep(self.poll_interval)


class ResumableLLMServerClient(LLMServerClient):
    """LLM client that hides rollout interruption from agent loops.

    Agent loops still await a single ``generate`` call. If colocated training
    closes the rollout gate and aborts in-flight vLLM requests, this client waits
    for the gate to reopen and re-issues the request with already returned tokens.
    """

    def __init__(
        self,
        config: DictConfig,
        load_balancer_handle: ray.actor.ActorHandle = None,
        rollout_gate: ray.actor.ActorHandle = None,
        gate_poll_interval: float = 1.0,
        **kwargs,
    ):
        super().__init__(config=config, load_balancer_handle=load_balancer_handle, **kwargs)
        self.rollout_gate = rollout_gate
        self.gate_poll_interval = gate_poll_interval

    async def _wait_for_rollout_gate(self) -> None:
        while self.rollout_gate is not None and not await self.rollout_gate.is_open.remote():
            await asyncio.sleep(self.gate_poll_interval)

    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Any = None,
        video_data: Any = None,
        audio_data: Any = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TokenOutput:
        original_max_tokens = sampling_params.get("max_tokens", sampling_params.get("max_new_tokens", None))
        if original_max_tokens is None:
            original_max_tokens = self.config.actor_rollout_ref.rollout.response_length
        original_max_tokens = int(original_max_tokens)

        token_ids: list[int] = []
        log_probs: list[float] | None = [] if sampling_params.get("logprobs") is not None else None
        routed_experts = None
        num_preempted = 0
        stop_reason = None
        min_global_steps = None
        max_global_steps = None
        segment_count = 0
        segment_token_counts: list[int] = []
        extra_fields: dict[str, Any] = {}

        while True:
            await self._wait_for_rollout_gate()

            run_sampling_params = dict(sampling_params)
            remaining_tokens = original_max_tokens - len(token_ids)
            if remaining_tokens <= 0:
                stop_reason = "length"
                break
            run_sampling_params.pop("max_new_tokens", None)
            run_sampling_params["max_tokens"] = remaining_tokens

            output = await super().generate(
                request_id=request_id,
                prompt_ids=prompt_ids + token_ids,
                sampling_params=run_sampling_params,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
                **kwargs,
            )
            segment_count += 1
            output_token_count = len(output.token_ids or [])
            segment_token_counts.append(output_token_count)

            if output.token_ids:
                token_ids.extend(output.token_ids)
            if log_probs is not None:
                if output_token_count == 0:
                    if output.log_probs not in (None, []):
                        raise RuntimeError(
                            "Rollout logprob/token length mismatch: "
                            f"logprobs={len(output.log_probs)} tokens=0"
                        )
                elif output.log_probs is None:
                    raise RuntimeError("Rollout logprobs are required but the vLLM segment returned none.")
                elif len(output.log_probs) != output_token_count:
                    raise RuntimeError(
                        "Rollout logprob/token length mismatch: "
                        f"logprobs={len(output.log_probs)} tokens={output_token_count}"
                    )
                elif output.log_probs:
                    log_probs.extend(output.log_probs)

            if output.routed_experts is not None and output.token_ids:
                if routed_experts is None:
                    routed_experts = output.routed_experts
                else:
                    routed_experts = torch.cat(
                        [routed_experts, output.routed_experts[-len(output.token_ids) :]],
                        dim=0,
                    )

            if output.num_preempted is not None:
                num_preempted += output.num_preempted

            if output.extra_fields:
                extra_fields.update(output.extra_fields)
                global_steps = output.extra_fields.get("global_steps")
                if global_steps is not None:
                    min_global_steps = global_steps if min_global_steps is None else min(min_global_steps, global_steps)
                    max_global_steps = global_steps if max_global_steps is None else max(max_global_steps, global_steps)

            stop_reason = output.stop_reason
            if len(token_ids) >= original_max_tokens:
                stop_reason = "length"
                break
            if stop_reason not in ("aborted", "abort"):
                break

            await asyncio.sleep(self.gate_poll_interval)

        if min_global_steps is not None:
            extra_fields["min_global_steps"] = min_global_steps
            extra_fields["max_global_steps"] = max_global_steps
            extra_fields["global_steps"] = max_global_steps
        extra_fields["partial_rollout_segments"] = segment_count
        extra_fields["partial_rollout_resumed"] = segment_count > 1
        extra_fields["partial_rollout_segment_token_counts"] = segment_token_counts
        extra_fields["partial_rollout_segment_token_counts_str"] = ",".join(
            str(count) for count in segment_token_counts
        )
        extra_fields["partial_rollout_total_response_tokens"] = len(token_ids)
        extra_fields["partial_rollout_token_digest"] = token_ids_digest(token_ids)

        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            routed_experts=routed_experts,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
            extra_fields=extra_fields,
        )


def compute_advantage_for_multi_trajectories(
    data: DataProto,
    batch_keys: list[str],
    adv_estimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Any = None,
) -> DataProto:
    """Compute GRPO advantages from each session's final output. For non-GRPO
    estimators, such as GAE, are delegated to the original compute_advantage() unchanged.

    For GRPO, only the final output in each ``{uid}_{session_id}`` group participates
    in advantage computation, and the result is broadcast to the other outputs in
    the same session. Sessions whose AgentLoop returns ``None`` simply do not appear
    in ``batch_keys``. Non-GRPO estimators, such as GAE, are delegated to the
    original ``compute_advantage()`` unchanged.
    """
    if adv_estimator != core_algos.AdvantageEstimator.GRPO:
        result = compute_advantage(
            data,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )
        return _add_process_advantages(result)

    # final session of each agent loop: {uid}_{session_id} => (index, row_index)
    final_sessions: dict[str, tuple[int, int]] = {}
    row_session_keys = []
    for i, key in enumerate(batch_keys):
        fields = key.rsplit("_", 2)
        assert len(fields) == 3, f"Unexpected key format: {key}"
        uid, session_id, index = fields[0], fields[1], int(fields[2])
        session_key = f"{uid}_{session_id}"
        row_session_keys.append(session_key)
        if session_key not in final_sessions or final_sessions[session_key][0] < index:
            final_sessions[session_key] = (index, i)

    # final session indices in batch data
    final_indices = []
    session_key_to_local_index = {}
    for session_key, (_, row_index) in final_sessions.items():
        final_indices.append(row_index)
        session_key_to_local_index[session_key] = len(final_indices) - 1
    row_to_local_index = [session_key_to_local_index[session_key] for session_key in row_session_keys]

    # select final sessions from batch data for group relative advantage computation
    final_data = compute_advantage(
        data.select_idxs(final_indices),
        adv_estimator=adv_estimator,
        gamma=gamma,
        lam=lam,
        num_repeat=num_repeat,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )
    first_nnz_indices = final_data.batch["response_mask"].argmax(dim=1)
    final_scores = final_data.batch["advantages"][torch.arange(len(final_data)), first_nnz_indices]

    # scatter final scores to all rows in batch data
    scores = final_scores[row_to_local_index]
    scores = scores.unsqueeze(-1) * data.batch["response_mask"]

    data.batch["advantages"] = scores
    data.batch["returns"] = scores
    return _add_process_advantages(data)


def _add_process_advantages(data: DataProto) -> DataProto:
    """Add action-local process signals without changing outcome-reward grouping."""
    if "process_advantages" not in data.batch:
        return data
    process_advantages = data.batch["process_advantages"] * data.batch["response_mask"]
    data.batch["advantages"] = data.batch["advantages"] + process_advantages
    data.batch["returns"] = data.batch["returns"] + process_advantages
    return data


class ReplayBuffer:
    """Replay buffer periodically polls metadata from transfer queue.

    Args:
        poll_interval (float, optional): Poll interval in seconds. Defaults to 1.0.
    """

    def __init__(self, poll_interval: float = 1.0):
        # partition_id => {key: tags}
        self.partitions: dict[str, dict[str, dict]] = defaultdict(dict)

        self.poll_interval = poll_interval
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self.poll_thread = threading.Thread(target=self._poll_from_transfer_queue, daemon=True)
        self.poll_thread.start()

    def _poll_from_transfer_queue(self):
        """Periodically poll metadata from transfer queue."""
        try:
            while not self._stop_event.is_set():
                data = tq.kv_list()
                if data is not None:
                    for partition_id, items in data.items():
                        self.add(partition_id, items)
                self._stop_event.wait(self.poll_interval)
        except Exception as e:
            if not self._stop_event.is_set():
                logger.error(f"Error in _poll_from_transfer_queue: {e}")
                os._exit(1)

    def close(self):
        """Stop the background polling thread."""
        if not self.poll_thread.is_alive():
            return
        self._stop_event.set()
        self.poll_thread.join(timeout=self.poll_interval + 1.0)
        if self.poll_thread.is_alive():
            logger.warning("ReplayBuffer poll thread did not stop within timeout")

    def add(self, partition_id: str, items: dict[str, dict]):
        """Add items to the replay buffer.

        Args:
            partition_id (str): Partition of transfer queue, e.g. "train" or "val".
            items (dict[str, dict]): Items to add, e.g. {"key": {"tag": "value"}}.
        """
        with self.lock:
            partition = self.partitions[partition_id]
            for key, tags in items.items():
                if key not in partition:
                    partition[key] = {}
                partition[key].update(tags)

    def remove(self, partition_id: str, keys: list[str]):
        """Remove items from the replay buffer.

        Args:
            partition_id (str): Partition of transfer queue, e.g. "train" or "val".
            keys (list[str]): Keys to remove.
        """
        with self.lock:
            partition = self.partitions[partition_id]
            for key in keys:
                if key in partition:
                    del partition[key]

    @staticmethod
    def _parse_output_key(key: str) -> tuple[str, int, int] | None:
        parts = str(key).rsplit("_", 2)
        if len(parts) != 3:
            return None
        uid, session_id, output_index = parts
        try:
            return uid, int(session_id), int(output_index)
        except ValueError:
            return None

    @staticmethod
    def _tag_int(tag: dict, key: str, default: int) -> int:
        value = tag.get(key, default)
        if hasattr(value, "item"):
            value = value.item()
        return int(value)

    def count_prompt_groups(self, partition_id: str, *, default_rollout_n: int) -> dict[str, int]:
        """Return prompt-group counts currently tracked in the replay buffer."""
        with self.lock:
            partition = self.partitions[partition_id]
            success_by_uid: dict[str, set[int]] = defaultdict(set)
            for key, tag in partition.items():
                if tag.get("status") != "success":
                    continue
                parsed = self._parse_output_key(key)
                if parsed is None:
                    continue
                uid, session_id, _ = parsed
                success_by_uid[uid].add(session_id)

            counts = {
                "running": 0,
                "finished_ready": 0,
                "finished_incomplete": 0,
                "failure": 0,
                "outstanding": 0,
            }
            for key, tag in partition.items():
                status = tag.get("status")
                if status == "running":
                    counts["running"] += 1
                elif status == "finished":
                    expected = self._tag_int(tag, "expected_rollout_n", default_rollout_n)
                    if len(success_by_uid.get(key, set())) >= expected:
                        counts["finished_ready"] += 1
                    else:
                        counts["finished_incomplete"] += 1
                elif status == "failure":
                    counts["failure"] += 1

            counts["outstanding"] = counts["running"] + counts["finished_ready"] + counts["finished_incomplete"]
            return counts

    def sample_ready_prompt_groups(
        self,
        partition_id: str,
        *,
        current_prompt_keys: list[str],
        current_global_steps: int,
        min_current_prompt_groups: int,
        max_off_policy_steps: int | None,
        default_rollout_n: int,
        ready_grace_wait_seconds: float = 0.0,
    ) -> KVBatchMeta:
        """Sample complete prompt groups with grouped-partial rollout semantics.

        ``min_current_prompt_groups`` applies only to prompts submitted in the
        current step. Complete historical prompt groups are appended to the train
        batch, but they never count toward the current-step ready ratio. Historical
        prompt groups older than ``max_off_policy_steps`` block the sampler until
        ready so the buffer cannot drift indefinitely.
        """
        current_prompt_keys = [str(key) for key in current_prompt_keys]
        current_prompt_key_set = set(current_prompt_keys)
        assert current_prompt_keys, "current_prompt_keys must be non-empty"
        assert min_current_prompt_groups > 0, "min_current_prompt_groups must be positive"
        assert min_current_prompt_groups <= len(current_prompt_keys), (
            "min_current_prompt_groups must be <= current prompt count"
        )

        ready_grace_wait_seconds = max(0.0, float(ready_grace_wait_seconds))
        grace_start_time: float | None = None
        grace_deadline: float | None = None
        grace_ready_at_start = 0
        grace_wait_polls = 0
        grace_timeout_reached = False
        stale_wait_polls = 0
        max_stale_wait_prompt_groups = 0
        while True:
            time.sleep(self.poll_interval)
            with self.lock:
                partition = self.partitions[partition_id]
                success_by_uid: dict[str, list[tuple[int, int, str, dict]]] = defaultdict(list)
                for key, tag in partition.items():
                    if tag.get("status") != "success":
                        continue
                    parsed = self._parse_output_key(key)
                    if parsed is None:
                        continue
                    uid, session_id, output_index = parsed
                    success_by_uid[uid].append((session_id, output_index, key, tag))

                current_ready_uids: list[str] = []
                historical_ready_uids: list[str] = []
                stale_wait_uids: list[str] = []
                running_prompt_groups = 0
                finished_prompt_groups = 0
                failure_prompt_groups = 0
                for key, tag in partition.items():
                    status = tag.get("status")
                    if status == "running":
                        running_prompt_groups += 1
                    elif status == "finished":
                        finished_prompt_groups += 1
                    elif status == "failure":
                        failure_prompt_groups += 1
                    else:
                        continue

                    expected_rollout_n = self._tag_int(tag, "expected_rollout_n", default_rollout_n)
                    session_ids = {session_id for session_id, _, _, _ in success_by_uid.get(key, [])}
                    is_ready = status == "finished" and len(session_ids) >= expected_rollout_n
                    if is_ready:
                        if key in current_prompt_key_set:
                            current_ready_uids.append(key)
                        else:
                            historical_ready_uids.append(key)
                        continue

                    prompt_global_steps = self._tag_int(tag, "global_steps", current_global_steps)
                    is_historical = key not in current_prompt_key_set and prompt_global_steps < current_global_steps
                    if (
                        max_off_policy_steps is not None
                        and max_off_policy_steps > 0
                        and is_historical
                        and status != "failure"
                        and current_global_steps - prompt_global_steps >= max_off_policy_steps
                    ):
                        stale_wait_uids.append(key)

                if stale_wait_uids:
                    stale_wait_polls += 1
                    max_stale_wait_prompt_groups = max(max_stale_wait_prompt_groups, len(stale_wait_uids))
                    continue

                current_ready_uid_set = set(current_ready_uids)
                current_ready_uids = [key for key in current_prompt_keys if key in current_ready_uid_set]
                if len(current_ready_uids) < min_current_prompt_groups:
                    continue
                if ready_grace_wait_seconds > 0 and len(current_ready_uids) < len(current_prompt_keys):
                    now = time.monotonic()
                    if grace_deadline is None:
                        grace_start_time = now
                        grace_deadline = now + ready_grace_wait_seconds
                        grace_ready_at_start = len(current_ready_uids)
                    if now < grace_deadline:
                        grace_wait_polls += 1
                        continue
                    grace_timeout_reached = True

                historical_ready_uids = sorted(
                    historical_ready_uids,
                    key=lambda uid: (
                        self._tag_int(partition[uid], "global_steps", current_global_steps),
                        uid,
                    ),
                )
                selected_uids = historical_ready_uids + current_ready_uids
                keys: list[str] = []
                tags: list[dict] = []
                selected_success_sessions: list[int] = []
                selected_expected_sessions: list[int] = []
                for uid in selected_uids:
                    group_items = sorted(success_by_uid[uid], key=lambda item: (item[0], item[1], item[2]))
                    keys.extend(item[2] for item in group_items)
                    tags.extend(item[3] for item in group_items)
                    selected_success_sessions.append(len({item[0] for item in group_items}))
                    selected_expected_sessions.append(
                        self._tag_int(partition[uid], "expected_rollout_n", default_rollout_n)
                    )

                missing_rollout_groups = sum(
                    success < expected
                    for success, expected in zip(selected_success_sessions, selected_expected_sessions, strict=True)
                )
                extra_success_groups = sum(
                    success > expected
                    for success, expected in zip(selected_success_sessions, selected_expected_sessions, strict=True)
                )
                min_success_sessions = min(selected_success_sessions, default=0)
                max_success_sessions = max(selected_success_sessions, default=0)
                grace_waited_seconds = (
                    max(0.0, time.monotonic() - grace_start_time) if grace_start_time is not None else 0.0
                )
                current_added_by_grace = (
                    max(0, len(current_ready_uids) - grace_ready_at_start) if grace_start_time is not None else 0
                )

                return KVBatchMeta(
                    partition_id=partition_id,
                    keys=keys,
                    tags=tags,
                    extra_info={
                        "grouped_partial/prompt_keys": selected_uids,
                        "grouped_partial/current_prompt_groups": len(current_prompt_keys),
                        "grouped_partial/current_min_prompt_groups": min_current_prompt_groups,
                        "grouped_partial/current_ready_prompt_groups": len(current_ready_uids),
                        "grouped_partial/current_selected_prompt_groups": len(current_ready_uids),
                        "grouped_partial/ready_grace_wait_seconds": ready_grace_wait_seconds,
                        "grouped_partial/ready_grace_waited_seconds": grace_waited_seconds,
                        "grouped_partial/ready_grace_wait_polls": grace_wait_polls,
                        "grouped_partial/ready_grace_timeout_reached": int(grace_timeout_reached),
                        "grouped_partial/current_ready_prompt_groups_at_grace_start": grace_ready_at_start,
                        "grouped_partial/current_ready_prompt_groups_added_by_grace": current_added_by_grace,
                        "grouped_partial/historical_ready_prompt_groups": len(historical_ready_uids),
                        "grouped_partial/historical_selected_prompt_groups": len(historical_ready_uids),
                        "grouped_partial/ready_prompt_groups": len(current_ready_uids) + len(historical_ready_uids),
                        "grouped_partial/selected_prompt_groups": len(selected_uids),
                        "grouped_partial/selected_output_rows": len(keys),
                        "grouped_partial/selected_min_success_sessions": min_success_sessions,
                        "grouped_partial/selected_max_success_sessions": max_success_sessions,
                        "grouped_partial/selected_prompt_groups_missing_rollouts": missing_rollout_groups,
                        "grouped_partial/selected_prompt_groups_extra_rollouts": extra_success_groups,
                        "grouped_partial/running_prompt_groups": running_prompt_groups,
                        "grouped_partial/finished_prompt_groups": finished_prompt_groups,
                        "grouped_partial/failure_prompt_groups": failure_prompt_groups,
                        "grouped_partial/stale_wait_prompt_groups": len(stale_wait_uids),
                        "grouped_partial/stale_wait_polls": stale_wait_polls,
                        "grouped_partial/max_stale_wait_prompt_groups": max_stale_wait_prompt_groups,
                        "grouped_partial/max_off_policy_steps": max_off_policy_steps or 0,
                    },
                )

    def sample(self, partition_id: str, global_steps: int = None, batch_size: int = None) -> KVBatchMeta:
        """Sample a batch of data from the replay buffer.

        Args:
            partition_id (str): Partition of transfer queue, e.g. "train" or "val".
            global_steps (int, optional): Global training steps. If not None, wait until all prompts of
                this global steps have finished.
            batch_size (int, optional): Batch size. Defaults to None.

        Returns:
            KVBatchMeta: A batch of data.
        """
        assert (global_steps is not None or batch_size) and (not (global_steps is not None and batch_size)), (
            "Either global_steps or batch_size must be specified, but not both."
        )

        while True:
            time.sleep(self.poll_interval)
            with self.lock:
                keys, tags = [], []
                should_wait = False
                partition = self.partitions[partition_id]
                for key, tag in partition.items():
                    if tag["global_steps"] == global_steps:
                        if tag["status"] == "running":
                            should_wait = True
                            break
                        elif tag["status"] == "success":
                            keys.append(key)
                            tags.append(tag)
                        else:
                            logger.debug(f"Unknown status {tag['status']} for key {key}")
                if not should_wait:
                    return KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags)


@ray.remote
class AgentLoopWorkerTQ(AgentLoopWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()
        self.background_tasks = set()
        max_concurrent = self.config.actor_rollout_ref.rollout.agent.get("max_concurrent_rollouts", None)
        if max_concurrent is None:
            self.rollout_semaphore = None
        else:
            num_workers = max(int(self.config.actor_rollout_ref.rollout.agent.num_workers), 1)
            per_worker = max(math.ceil(int(max_concurrent) / num_workers), 1)
            self.rollout_semaphore = asyncio.Semaphore(per_worker)

    def _compute_multi_modal_inputs(self, output, input_ids) -> dict[str, torch.Tensor]:
        multi_modal_data = output.multi_modal_data or {}
        has_multimodal_data = any(multi_modal_data.get(key) is not None for key in ("images", "videos", "audios"))
        if not has_multimodal_data:
            return {}

        return super()._compute_multi_modal_inputs(output, input_ids)

    def _compute_position_ids(
        self,
        input_ids,
        attention_mask,
        multi_modal_inputs,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        has_multimodal_positions = bool(multi_modal_inputs) and (
            multi_modal_inputs.get("image_grid_thw") is not None or multi_modal_inputs.get("video_grid_thw") is not None
        )
        if not has_multimodal_positions:
            return compute_position_id_with_mask(attention_mask)

        return super()._compute_position_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            multi_modal_inputs=multi_modal_inputs,
            mm_processor_kwargs=mm_processor_kwargs,
        )

    async def generate_sequences(self, batch: TensorDict) -> None:
        """Spawn agent loop for each sample in the batch without waiting for the results."""
        validate = batch["validate"] if "validate" in batch else False
        batch.pop("validate", None)
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch:
            default_agent_loop = config.agent.default_agent_loop
            batch["agent_name"] = NonTensorData(default_agent_loop)

        trajectory_info = await get_trajectory_info(batch["global_steps"], batch["index"], validate)

        # create background tasks for each sample in the batch
        for i in range(len(batch)):
            # TODO(wuxibin): add trace support
            trace_this_sample = False
            prompt = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    prompt[k] = v[i]
                elif isinstance(v, NonTensorStack):
                    prompt[k] = v[i].data
                elif isinstance(v, NonTensorData):
                    prompt[k] = v.data
                else:
                    logger.exception(f"Unsupported type {type(v)} for key {k}")

            # “fire-and-forget” background tasks
            task = asyncio.create_task(
                self._run_prompt(prompt, sampling_params, trajectory=trajectory_info[i], trace=trace_this_sample)
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def cancel_background_tasks(self) -> None:
        """Cancel unfinished agent-loop tasks during trainer shutdown."""
        tasks = list(self.background_tasks)
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.background_tasks.clear()

    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False) -> None:
        """Spawn multiple agent loops in parallel according to rollout.n or rollout.val_kwargs.n."""
        uid, partition_id = str(prompt["uid"]), "train" if not trajectory["validate"] else "val"
        prompt_tag = {
            "global_steps": int(trajectory.get("global_steps", prompt.get("global_steps", -1))),
            "expected_rollout_n": int(
                self.config.actor_rollout_ref.rollout.n
                if not trajectory["validate"]
                else self.config.actor_rollout_ref.rollout.val_kwargs.n
            ),
        }
        try:
            # NOTE: user can dynamically adjust n for each sample here, e.g according to task difficulty.
            config = self.config.actor_rollout_ref.rollout
            n = int(prompt.pop("__rollout_n__", config.n if not trajectory["validate"] else config.val_kwargs.n))
            do_sample = prompt.pop("__do_sample__", True)
            prompt_tag["expected_rollout_n"] = n

            run_sampling_params = dict(sampling_params)
            if not trajectory["validate"] and not do_sample:
                apply_greedy_sampling_params(run_sampling_params)

            async def run_session(session_id: int) -> None:
                sample_prompt = dict(prompt)
                sample_prompt["gen_uid"] = str(uuid.uuid4())
                if self.rollout_semaphore is None:
                    await self._run_agent_loop(
                        run_sampling_params,
                        trajectory=trajectory,
                        trace=trace,
                        session_id=session_id,
                        **sample_prompt,
                    )
                else:
                    async with self.rollout_semaphore:
                        await self._run_agent_loop(
                            run_sampling_params,
                            trajectory=trajectory,
                            trace=trace,
                            session_id=session_id,
                            **sample_prompt,
                        )

            tasks = []
            for i in range(n):
                task = asyncio.create_task(run_session(i))
                tasks.append(task)
            await asyncio.gather(*tasks)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={**prompt_tag, "status": "finished"})
        except Exception as e:
            logger.exception(f"Error in _run_prompt: {e}")
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={**prompt_tag, "status": "failure"})

    async def _agent_loop_postprocess(
        self, output: AgentLoopOutput | list[AgentLoopOutput], validate, **kwargs
    ) -> None:
        """Put agent loop outputs into TransferQueue."""
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            logger.warning(f"Empty output for prompt {uid}_{session_id}")
            return

        await self._compute_score(outputs, kwargs=kwargs)

        final_output = outputs[-1]
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )

        if final_output.reward_score is not None:
            for output in outputs[:-1]:
                if output.reward_score is None:
                    output.reward_score = final_output.reward_score
                    output.extra_fields["reward_extra_info"] = final_output.extra_fields.get("reward_extra_info", {})

        # NOTE: agent loop may has multiple outputs, put each output into TransferQueue.
        # key format: {uid}_{session_id}_{index}
        # - uid: raw prompt uid from dataset
        # - session_id: session id for rollout.n sampling
        # - index: index of agent loop output
        keys, fields, tags = [], [], []
        for i, output in enumerate(outputs):
            prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(output.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)

            keys.append(f"{uid}_{session_id}_{i}")
            field = output.as_dict()
            field.update(kwargs)
            for extra_key in ("gen_uid", "agent_role"):
                if extra_key in output.extra_fields:
                    field[extra_key] = output.extra_fields[extra_key]
            # do not store raw image/video
            field.pop("multi_modal_data", None)
            # TODO: uniform response_mask and loss_mask
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            field["multi_modal_inputs"] = multi_modal_inputs
            fields.append(field)
            prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
            tags.append(
                {
                    "global_steps": kwargs["global_steps"],
                    "status": "success",
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "seq_len": prompt_len + response_len,
                }
            )

        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="train" if not validate else "val",
        )


class AgentLoopManagerTQ(AgentLoopManager):
    def __init__(self, *args, replay_buffer: ReplayBuffer, **kwargs):
        self.agent_loop_workers_class = AgentLoopWorkerTQ
        super().__init__(*args, **kwargs)
        self.replay_buffer = replay_buffer

    @classmethod
    @auto_await
    async def create(
        cls,
        *args,
        replay_buffer: ReplayBuffer = None,
        **kwargs,
    ):
        """Create agent loop manager."""
        instance = cls(
            *args,
            **kwargs,
            replay_buffer=replay_buffer,
        )
        await instance._init_agent_loop_workers()
        return instance

    def generate_sequences(self, prompts: TensorDict) -> None:
        """
        Dispatch input batch to agent loop workers without blocking. Workers should put agent loop outputs
        into TransferQueue once an agent loop finished.

        Args:
            prompts (TensorDict): Input batch from train or validation dataset.
        """
        # mark prompts as pending in replay buffer
        global_steps = prompts["global_steps"]
        if isinstance(global_steps, NonTensorData):
            global_steps = global_steps.data
        if hasattr(global_steps, "item"):
            global_steps = global_steps.item()
        global_steps = int(global_steps)
        partition_id = "train" if "validate" not in prompts else "val"
        rollout_n_override = prompts.get("__rollout_n__", None)
        items = {}
        for i, uid in enumerate(prompts["uid"]):
            uid = str(uid)
            expected_rollout_n = (
                int(rollout_n_override[i].item() if hasattr(rollout_n_override[i], "item") else rollout_n_override[i])
                if rollout_n_override is not None
                else int(
                    self.config.actor_rollout_ref.rollout.n
                    if partition_id == "train"
                    else self.config.actor_rollout_ref.rollout.val_kwargs.n
                )
            )
            items[uid] = {
                "global_steps": global_steps,
                "expected_rollout_n": expected_rollout_n,
                "status": "running",
            }
        self.replay_buffer.add(partition_id, items)

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=False)
            ]
        )

    def cancel_background_tasks(self) -> None:
        ray.get([worker.cancel_background_tasks.remote() for worker in self.agent_loop_workers])


# ======================================= USER SECTION END =======================================


class PPOTrainer:
    """PPO Trainer with TransferQueue and ReplayBuffer.

    Args:
        config: DictConfig from yaml config file.
        role_worker_mapping: dict[Role, WorkerType]
        resource_pool_manager: ResourcePoolManager
    """

    def __init__(
        self,
        config: DictConfig,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
    ):
        self.config = config
        self._normalize_rollout_logprob_config()
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_critic = need_critic(self.config)
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_teacher_policy = need_teacher_policy(self.config)
        self.replay_buffer = ReplayBuffer()
        self._curriculum_success_ema: dict[str, float] = {}
        self._curriculum_observations: defaultdict[str, int] = defaultdict(int)
        self.grouped_partial_enabled = bool(
            OmegaConf.select(self.config, "trainer.grouped_partial_rollout.enable", default=False)
        )
        self.rollout_gate = None
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._init_tokenizer()
        self._init_dataloader()
        self._init_dump_executor()

    def _normalize_rollout_logprob_config(self) -> None:
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        if not (rollout_corr_config and rollout_corr_config.get("bypass_mode", False)):
            return

        with open_dict(self.config):
            self.config.actor_rollout_ref.rollout.calculate_log_probs = True
            self.config.actor_rollout_ref.actor.use_rollout_log_probs = True
            self.config.actor_rollout_ref.actor.policy_loss.loss_mode = "bypass_mode"
            self.config.actor_rollout_ref.actor.policy_loss.rollout_correction = OmegaConf.to_container(
                rollout_corr_config,
                resolve=True,
            )

    def _grouped_partial_gate_poll_interval(self) -> float:
        return float(
            OmegaConf.select(
                self.config,
                "trainer.grouped_partial_rollout.gate_poll_interval",
                default=0.25,
            )
        )

    def _grouped_partial_ready_grace_wait_seconds(self) -> float:
        return float(
            OmegaConf.select(
                self.config,
                "trainer.grouped_partial_rollout.ready_grace_wait_seconds",
                default=5.0,
            )
        )

    def _grouped_partial_target_prompt_groups(self) -> int:
        return int(
            OmegaConf.select(
                self.config,
                "trainer.grouped_partial_rollout.target_prompt_groups",
                default=self.config.data.train_batch_size,
            )
        )

    def _grouped_partial_min_prompt_groups(self, current_prompt_groups: int | None = None) -> int:
        explicit_min = OmegaConf.select(
            self.config,
            "trainer.grouped_partial_rollout.min_prompt_groups",
            default=None,
        )
        target_prompt_groups = current_prompt_groups or self._grouped_partial_target_prompt_groups()
        if explicit_min is not None:
            return max(1, min(int(explicit_min), target_prompt_groups))

        ready_ratio = float(
            OmegaConf.select(
                self.config,
                "trainer.grouped_partial_rollout.ready_ratio",
                default=1.0,
            )
        )
        return max(1, math.ceil(target_prompt_groups * ready_ratio))

    def _grouped_partial_max_off_policy_steps(self) -> int:
        return int(
            OmegaConf.select(
                self.config,
                "trainer.grouped_partial_rollout.max_off_policy_steps",
                default=5,
            )
        )

    def _open_rollout_gate(self) -> None:
        if self.rollout_gate is not None:
            ray.get(self.rollout_gate.open.remote())

    def _close_rollout_gate(self) -> None:
        if self.rollout_gate is not None:
            ray.get(self.rollout_gate.close.remote())

    def _resume_rollout_generation(self) -> None:
        if self.grouped_partial_enabled:
            self.checkpoint_manager.resume_generation_replicas()

    def _init_tokenizer(self):
        """Initialize tokenizer."""
        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            self.config.actor_rollout_ref.model.path, use_shm=self.config.actor_rollout_ref.model.get("use_shm", False)
        )
        trust_remote_code = self.config.data.get("trust_remote_code", False)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    def _init_dataloader(self):
        """Initialize train and validate dataloader."""
        self.train_dataset = create_rl_dataset(
            self.config.data.train_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            is_train=True,
            max_samples=self.config.data.get("train_max_samples", -1),
        )
        self.val_dataset = create_rl_dataset(
            self.config.data.val_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            is_train=False,
            max_samples=self.config.data.get("val_max_samples", -1),
        )

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data["dataloader_num_workers"],
            drop_last=True,
            collate_fn=collate_fn,
            sampler=create_rl_sampler(self.config.data, self.train_dataset),
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.val_batch_size or len(self.val_dataset),
            num_workers=self.config.data["dataloader_num_workers"],
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )
        logger.info(
            f"train and validate dataloader initialized, train dataset size: "
            f"{len(self.train_dataset)}, val dataset size: {len(self.val_dataset)}"
        )

        # adjust total_training_steps
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        logger.info(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            logger.warning(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # 1. define actor and rollout class
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[actor_role],
            config=self.config.actor_rollout_ref,
            distillation_config=self.config.get("distillation"),
            role=str(actor_role),
        )
        self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls

        # 2. define critic class
        if self.use_critic:
            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)
            critic_cfg.engine.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            critic_cfg.engine.max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            worker_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=critic_cfg.model_config,
                engine_config=critic_cfg.engine,
                optimizer_config=critic_cfg.optim,
                checkpoint_config=critic_cfg.checkpoint,
            )
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=worker_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # 3. create worker group for actor rollout and critic
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.config.trainer.device
        logger.info(f"worker group kwargs: {wg_kwargs}")

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = RayWorkerGroup(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            logger.info(f"create worker group {spawn_wg.keys()}")

        # 5. initialize critic model engine
        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            value_loss_ = partial(value_loss, config=critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)
            logger.info("critic model engine initialized")

        # 6. initialize actor and ref model engine
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()
        logger.info("actor and ref model engine initialized")

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = self.config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = self.config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or self.config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker already hosts the reference model.
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]
        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # 7. initialize reward loop manager
        resource_pool = (
            self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            if self.config.reward.reward_model.enable
            else None
        )
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )
        logger.info("reward loop manager initialized")

        # 8. initialize teacher loop manager
        if self.use_teacher_policy:
            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
            self.teacher_model_manager = MultiTeacherModelManager(
                config=self.config,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        # 9. initialize agent loop manager
        self.llm_server_manager = LLMServerManager.create(
            config=self.config, worker_group=self.actor_rollout_wg, rollout_resource_pool=actor_rollout_resource_pool
        )
        if self.grouped_partial_enabled:
            self.rollout_gate = RolloutGate.remote(
                opened=True,
                poll_interval=self._grouped_partial_gate_poll_interval(),
            )
            llm_client = self.llm_server_manager.get_client(
                client_cls=ResumableLLMServerClient,
                rollout_gate=self.rollout_gate,
                gate_poll_interval=self._grouped_partial_gate_poll_interval(),
            )
        else:
            llm_client = self.llm_server_manager.get_client()

        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            agent_loop_manager_cls = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            agent_loop_manager_cls = AgentLoopManagerTQ
        self.async_rollout_manager = agent_loop_manager_cls.create(
            config=self.config,
            llm_client=llm_client,
            teacher_client=self.teacher_model_manager.get_client() if self.use_teacher_policy else None,
            reward_loop_worker_handles=self.reward_loop_manager.reward_loop_workers,
            replay_buffer=self.replay_buffer,
        )
        logger.info("agent loop manager initialized")

        # 10. initialize checkpoint engine manager
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )
        logger.info("checkpoint engine manager initialized")

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

        logger.info("all initialize finished, ready to fit")

    def _load_checkpoint(self):
        self.global_steps = 0

        # 1. find latest checkpoint folder
        if self.config.trainer.resume_mode == "disable":
            return
        elif self.config.trainer.resume_mode == "auto":
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest
            if global_step_folder is None:
                logger.info("Training from scratch")
                return
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
            assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)
        else:
            logger.exception(f"Unknown resume mode {self.config.trainer.resume_mode}")

        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        logger.info(f"Resuming from {global_step_folder}, setting global step to {self.global_steps}")

        # 2. load actor checkpoint
        self.actor_rollout_wg.load_checkpoint(
            local_path=os.path.join(global_step_folder, "actor"),
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )

        # 3. load critic checkpoint
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                local_path=os.path.join(global_step_folder, str(Role.Critic)),
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )

        # 4. load dataloader checkpoint
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            logger.warning(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _save_checkpoint(self):
        """Save actor, critic, and dataloader checkpoints to local (and optionally remote) storage."""
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        logger.info(f"Saving checkpoint to {local_global_step_folder}")

        # resolve max checkpoints to keep
        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            logger.warning(
                "remove_previous_ckpt_in_save is deprecated, "
                "set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        # save actor
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        # save critic
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader state
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        torch.save(self.train_dataloader.state_dict(), dataloader_local_path)

        # write latest checkpointed iteration tracker for atomic resume
        actor_ckpt_cfg = self.config.actor_rollout_ref.actor.get("checkpoint", {})
        if actor_ckpt_cfg.get("async_save", False):
            logger.info("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _validate(self) -> dict[str, float]:
        # Lists to collect samples for the table
        sample_uids = []
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        data_sources = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        dump_all_inputs: list[str] = []
        dump_all_outputs: list[str] = []
        dump_all_keys: list[str] = []
        dump_all_gen_uids: list[str] = []
        gen_uid_to_sample_idx: dict[str, int] = {}

        for batch_dict in self.val_dataloader:
            # 1. put batch to agent loop manager
            batch_dict["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object
            )
            batch = tu.get_tensordict(batch_dict)
            tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
            tu.assign_non_tensor_data(batch, "validate", True)
            self.async_rollout_manager.generate_sequences(batch)

            # 2. sample batch from replay buffer
            batch = self.replay_buffer.sample(partition_id="val", global_steps=self.global_steps)

            # 3. [OPTIONAL] compute reward score with colocated reward model
            if self.reward_loop_manager.reward_loop_worker_handles is None:
                self.checkpoint_manager.sleep_replicas()
                batch = self._compute_reward_colocate(batch)
                self.checkpoint_manager.update_weights()

            # 4. collect necessary data for logging. Multi-output agent loops are
            # aggregated by gen_uid so a variable number of sub-sequences maps back
            # to one rollout slot for validation metrics.
            text_data = tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id, select_fields=["prompts", "responses"]
            )
            text_data["prompts"] = text_data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            text_data["responses"] = text_data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            all_inputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in text_data["prompts"]]
            all_outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in text_data["responses"]]

            fields = ["uid", "gen_uid", "rm_scores", "num_turns", "reward_model", "data_source", "extra_fields"]
            data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

            uids = data.pop("uid").tolist()
            gen_uids = [str(gen_uid) for gen_uid in data.pop("gen_uid").tolist()]
            scores = data["rm_scores"].sum(dim=1).tolist()
            num_turns = data.pop("num_turns").tolist()
            extra_fields_list = data.pop("extra_fields").tolist()

            reward_model = data.pop("reward_model", None)
            if reward_model is not None:
                ground_truths = [item.get("ground_truth", None) for item in reward_model.tolist()]
            else:
                ground_truths = [None] * len(batch.keys)

            data_source = data.pop("data_source", None)
            if data_source is not None:
                all_data_sources = data_source.tolist()
            else:
                all_data_sources = ["unknown"] * len(batch.keys)

            ordered_gen_uids: list[str] = []
            seen_gen_uids: set[str] = set()
            gen_uid_first_idx: dict[str, int] = {}
            gen_uid_score: dict[str, float] = {}
            for row_idx, gen_uid in enumerate(gen_uids):
                if gen_uid not in seen_gen_uids:
                    seen_gen_uids.add(gen_uid)
                    ordered_gen_uids.append(gen_uid)
                    gen_uid_first_idx[gen_uid] = row_idx
                    gen_uid_score[gen_uid] = scores[row_idx]
                else:
                    gen_uid_score[gen_uid] = max(gen_uid_score[gen_uid], scores[row_idx])

            base_offset = len(sample_scores)
            for local_idx, gen_uid in enumerate(ordered_gen_uids):
                first_idx = gen_uid_first_idx[gen_uid]
                score = gen_uid_score[gen_uid]
                gen_uid_to_sample_idx[gen_uid] = base_offset + local_idx
                sample_uids.append(uids[first_idx])
                sample_inputs.append(all_inputs[first_idx])
                sample_outputs.append(all_outputs[first_idx])
                sample_gts.append(ground_truths[first_idx])
                sample_scores.append(score)
                sample_turns.append(num_turns[first_idx])
                data_sources.append(all_data_sources[first_idx])

                reward_extra_info = (
                    extra_fields_list[first_idx].get("reward_extra_info", {})
                    if isinstance(extra_fields_list[first_idx], dict)
                    else {}
                )
                reward_extra_infos_dict["reward"].append(score)
                n_prior = len(reward_extra_infos_dict["reward"]) - 1
                for key in reward_extra_infos_dict:
                    if key != "reward" and key not in reward_extra_info:
                        reward_extra_infos_dict[key].append(None)
                for key, value in reward_extra_info.items():
                    if key == "reward":
                        continue
                    if key not in reward_extra_infos_dict:
                        reward_extra_infos_dict[key] = [None] * n_prior
                    reward_extra_infos_dict[key].append(value)

            dump_all_inputs.extend(all_inputs)
            dump_all_outputs.extend(all_outputs)
            dump_all_keys.extend(batch.keys)
            dump_all_gen_uids.extend(gen_uids)

            # 5. cleanup transfer queue and replay buffer
            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)
            self.replay_buffer.remove(batch.partition_id, batch.keys)

        # logger to wandb
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump to local dir
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            # Sort according to uid (so that generations in the same rollout are together)
            sort_keys = []
            for key in dump_all_keys:
                parts = key.rsplit("_", 2)
                sort_keys.append((parts[0], int(parts[1]), int(parts[2])) if len(parts) == 3 else (key, 0, 0))
            sorted_indices = sorted(range(len(dump_all_keys)), key=lambda i: sort_keys[i])
            dump_all_inputs = [dump_all_inputs[i] for i in sorted_indices]
            dump_all_outputs = [dump_all_outputs[i] for i in sorted_indices]
            dump_all_keys = [dump_all_keys[i] for i in sorted_indices]
            dump_all_gen_uids = [dump_all_gen_uids[i] for i in sorted_indices]

            # For ground truths, scores and reward extra infos, find the values
            # for the aggregated rollout slot for each raw output row.
            dump_sample_indices = [gen_uid_to_sample_idx[gen_uid] for gen_uid in dump_all_gen_uids]
            self._dump_generations(
                inputs=dump_all_inputs,
                outputs=dump_all_outputs,
                gts=[sample_gts[i] for i in dump_sample_indices],
                scores=[sample_scores[i] for i in dump_sample_indices],
                reward_extra_infos_dict={
                    k: [v[i] for i in dump_sample_indices] for k, v in reward_extra_infos_dict.items()
                }
                | {"uid": dump_all_keys, "gen_uid": dump_all_gen_uids},
                dump_path=val_data_dir,
            )

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""
        generations_to_log = self.config.trainer.log_val_generations
        if generations_to_log == 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    @staticmethod
    def _write_generations(inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, global_steps):
        """Write generation samples as JSONL (runs in background thread)."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        def json_encode_default(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, "tolist"):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False, default=json_encode_default) + "\n")

        print(f"Dumped generations to {filename}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL asynchronously."""
        global_steps = self.global_steps
        future = self._dump_executor.submit(
            self._write_generations,
            inputs,
            outputs,
            gts,
            scores,
            reward_extra_infos_dict,
            dump_path,
            global_steps,
        )
        self._dump_futures.append(future)
        # Clean up completed futures and surface any exceptions early
        still_pending = []
        for f in self._dump_futures:
            if f.done():
                f.result()  # re-raises if the write failed
            else:
                still_pending.append(f)
        self._dump_futures = still_pending

    def _init_dump_executor(self):
        """Create or recreate the dump executor and futures list."""
        self._dump_executor = ThreadPoolExecutor(max_workers=1)
        self._dump_futures = []

    def _shutdown_dump_executor(self):
        """Drain pending dump futures and shut down the executor."""
        for f in self._dump_futures:
            f.result()
        self._dump_futures.clear()
        self._dump_executor.shutdown(wait=True)

    def _log_rollout_data(self, batch: KVBatchMeta, timing_raw: dict, rollout_data_dir: str):
        """Fetch rollout data from TransferQueue and dump sorted by uid."""
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            fields = ["uid", "prompts", "responses", "rm_scores", "reward_model"]
            data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
            data["prompts"] = data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            data["responses"] = data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)

            uids = data.pop("uid").tolist()
            inputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in data["prompts"]]
            outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in data["responses"]]
            scores = data["rm_scores"].sum(dim=1).tolist()

            reward_model = data.pop("reward_model", None)
            if reward_model is not None:
                gts = [item.get("ground_truth", None) for item in reward_model.tolist()]
            else:
                gts = [None] * len(uids)

            # Sort by uid key ({sample}_{rollout}_{output})
            sort_keys = []
            for key in batch.keys:
                parts = key.rsplit("_", 2)
                if len(parts) == 3:
                    sort_keys.append((parts[0], int(parts[1]), int(parts[2])))
                else:
                    sort_keys.append((key, 0, 0))
            sorted_indices = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])

            inputs = [inputs[i] for i in sorted_indices]
            outputs = [outputs[i] for i in sorted_indices]
            gts = [gts[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]

            reward_extra_infos_dict = {"uid": [batch.keys[i] for i in sorted_indices]}

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=rollout_data_dir,
            )

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns) -> dict[str, float]:
        global_all_metrics = {}
        for var_name, values in reward_extra_infos_dict.items():
            if not (var_name == "all" or var_name.startswith("all/")):
                continue
            numeric_values = [
                value
                for value in values
                if value is not None and not isinstance(value, str)
            ]
            if numeric_values:
                global_all_metrics[var_name] = float(np.mean(numeric_values))

        filtered_infos_dict = {
            key: value
            for key, value in reward_extra_infos_dict.items()
            if not (key == "all" or key.startswith("all/"))
        }
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, filtered_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                if not metric2val:
                    continue
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        for var_name, metric_val in global_all_metrics.items():
            metric_dict[f"val-core/{var_name}"] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.array(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _start_profiling(self) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        do_profile = (
            not self.prev_step_profile and self.curr_step_profile
            if self.config.global_profiler.profile_continuous_steps
            else self.curr_step_profile
        )

        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        self.next_step_profile = (
            self.global_steps + 1 in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        do_profile = (
            self.curr_step_profile and not self.next_step_profile
            if self.config.global_profiler.profile_continuous_steps
            else self.curr_step_profile
        )
        self.prev_step_profile = self.curr_step_profile
        self.curr_step_profile = self.next_step_profile

        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _compute_reward_colocate(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the reward with colocate reward model."""
        # TODO: add reward model
        raise NotImplementedError

    def _add_remax_reward_baselines(self, batch: KVBatchMeta) -> KVBatchMeta:
        """Attach one greedy baseline reward to every sampled ReMax trajectory."""
        baseline_prefix = "remax_baseline_"
        sampled_keys, sampled_tags = [], []
        all_baseline_keys = []
        final_baseline_key_by_uid: dict[str, tuple[int, str]] = {}
        for key, tag in zip(batch.keys, batch.tags, strict=True):
            uid, _, index = key.rsplit("_", 2)
            if uid.startswith(baseline_prefix):
                all_baseline_keys.append(key)
                output_index = int(index)
                if uid not in final_baseline_key_by_uid or final_baseline_key_by_uid[uid][0] < output_index:
                    final_baseline_key_by_uid[uid] = (output_index, key)
            else:
                sampled_keys.append(key)
                sampled_tags.append(tag)

        assert final_baseline_key_by_uid, "ReMax requires greedy baseline rollout outputs, but none were found."
        baseline_keys = [key for _, key in final_baseline_key_by_uid.values()]
        baseline_data = tq.kv_batch_get(
            keys=baseline_keys, partition_id=batch.partition_id, select_fields=["uid", "rm_scores"]
        )
        baseline_scores = baseline_data["rm_scores"].sum(dim=-1)
        baseline_by_uid = {
            uid.removeprefix(baseline_prefix): score
            for uid, score in zip(list(baseline_data["uid"]), baseline_scores, strict=True)
        }

        sampled_data = tq.kv_batch_get(keys=sampled_keys, partition_id=batch.partition_id, select_fields=["uid"])
        reward_baselines = torch.stack([baseline_by_uid[uid] for uid in list(sampled_data["uid"])])
        tq.kv_batch_put(
            keys=sampled_keys,
            partition_id=batch.partition_id,
            fields=TensorDict({"reward_baselines": reward_baselines}, batch_size=len(sampled_keys)),
        )
        tq.kv_clear(keys=all_baseline_keys, partition_id=batch.partition_id)
        self.replay_buffer.remove(batch.partition_id, all_baseline_keys)
        return KVBatchMeta(
            keys=sampled_keys,
            tags=sampled_tags,
            partition_id=batch.partition_id,
            fields=batch.fields,
            extra_info=batch.extra_info,
        )

    def _get_required_batch_multiple(self, dp_size: int) -> int:
        """Return the global batch multiple required by downstream train steps(e.g. critics, actors)."""
        required_multiple = dp_size

        # If enabled with critic training, the batch should align with critic PPO mini-batches.
        if self.use_critic:
            critic_global_mini_batch_size = self.config.critic.ppo_mini_batch_size
            critic_global_mini_batch_size *= self.config.actor_rollout_ref.rollout.n
            required_multiple = math.lcm(required_multiple, critic_global_mini_batch_size)

        # If there is an actor update, the batch should align with actor PPO mini-batches too.
        if self.config.trainer.critic_warmup <= self.global_steps:
            actor_global_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            actor_global_mini_batch_size *= self.config.actor_rollout_ref.rollout.n
            required_multiple = math.lcm(required_multiple, actor_global_mini_batch_size)

        # Notice lcm(a, b, c) == lcm(lcm(a, b), c), so it is optimal.
        return required_multiple

    def _balance_batch(self, batch: KVBatchMeta, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens."""
        # get actor dp size
        role, worker_group = "actor", self.actor_rollout_wg
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        dp_size = max(dp_rank_mapping) + 1

        # Upsampling the batch with padding sequences
        batch_multiple = self._get_required_batch_multiple(dp_size)
        batch = upsample_batch_to_divisible_size(batch, batch_multiple, self.tokenizer.eos_token_id)
        global_seqlen_lst = torch.tensor([tag["seq_len"] for tag in batch.tags], dtype=torch.int64)
        workload_lst = calculate_workload(global_seqlen_lst)

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        batch.reorder([j for partition in global_partition_lst for j in partition])
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)
        return batch

    def _compute_old_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the old log prob of the batch."""
        # Operating Mode Selection:
        # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
        # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
        #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
            data = tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id, select_fields=["rollout_log_probs"]
            )
            data["old_log_probs"] = data.pop("rollout_log_probs")
            extra_info = dict(batch.extra_info)
            batch = tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data)
            batch.extra_info.update(extra_info)
            return batch

        # 1. compute log probs
        batch.extra_info.update(
            {
                "calculate_entropy": True,
                "compute_loss": False,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            }
        )
        output: KVBatchMeta = self.actor_rollout_wg.compute_log_prob(batch)
        assert len(output) == len(batch)

        fields = ["entropy", "log_probs", "response_mask"]
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            fields.extend(["responses", "rollout_log_probs"])
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        # 2. write old_log_probs and entropy back to TransferQueue
        data["old_log_probs"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
        data["entropy"] = response_from_nested(data.pop("entropy"), data["response_mask"])
        extra_info = dict(batch.extra_info)
        batch = tq.kv_batch_put(
            keys=batch.keys, partition_id=batch.partition_id, fields=data.select("old_log_probs", "entropy")
        )
        batch.extra_info.update(extra_info)

        data = DataProto(batch=data.to_padded_tensor())

        # 3. calculate actor entroy metrics
        actor_config = self.config.actor_rollout_ref.actor
        entropy_agg = agg_loss(
            loss_mat=data.batch["entropy"],
            loss_mask=data.batch["response_mask"],
            loss_agg_mode=actor_config.loss_agg_mode,
            loss_scale_factor=actor_config.loss_scale_factor,
        )
        old_log_prob_metrics = {
            "actor/entropy": entropy_agg.detach().item(),
            # "perf/mfu/actor_infer": old_log_prob_mfu,
        }
        metrics.update(old_log_prob_metrics)

        # 4. calculate rollout vs actor logprobs diff
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            metrics.update(
                compute_compact_offpolicy_metrics(
                    old_log_prob=data.batch["old_log_probs"],
                    rollout_log_prob=data.batch["rollout_log_probs"],
                    response_mask=data.batch["response_mask"].bool(),
                    prefix="actor/pre_update/",
                )
            )
            metrics.update(calculate_debug_metrics(data))

        return batch

    def _compute_ref_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the reference log prob of the batch."""
        # 1. compute log probs
        metadata = {
            "calculate_entropy": False,
            "compute_loss": False,
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
        }
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        batch.extra_info.update(metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.compute_log_prob(batch)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch)
        assert len(output) == len(batch)

        # 2. write ref_log_prob and entropy back to TransferQueue
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["log_probs", "response_mask"]
        )
        data["ref_log_prob"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select("ref_log_prob"))

        return batch

    def _compute_values(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the values of the batch."""
        # 1. compute value
        output = self.critic_wg.infer_batch(batch)
        # TODO: DataProtoFuture support KVBatchMeta
        ray.get(output.futures)

        # 2. write value back to TransferQueue
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["values", "response_mask"]
        )
        data["values"] = response_from_nested(data.pop("values"), data["response_mask"])
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select("values"))

        return batch

    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the advantage of the batch."""
        fields = ["uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs", "ref_log_prob", "values"]
        curriculum_config = self.config.algorithm.get("curriculum", None)
        curriculum_enabled = bool(curriculum_config and curriculum_config.get("enabled", False))
        curriculum_base_reward_field = None
        if curriculum_enabled:
            curriculum_base_reward_field = str(
                curriculum_config.get("base_reward_field", "agent/base_reward_score")
            )
            required_curriculum_fields = {
                "curriculum_task_id",
                "curriculum_weight",
                curriculum_base_reward_field,
            }
            available_fields = set(batch.fields or [])
            missing_fields = required_curriculum_fields - available_fields
            if missing_fields:
                raise ValueError(
                    "algorithm.curriculum.enabled requires top-level dataset fields: "
                    f"{sorted(missing_fields)}"
                )
            fields.extend(sorted(required_curriculum_fields))
        if batch.fields is not None and "process_advantages" in batch.fields:
            fields.append("process_advantages")
        if self.config.algorithm.adv_estimator in (
            core_algos.AdvantageEstimator.FOLDGRPO,
            core_algos.AdvantageEstimator.FOLDGRPO.value,
        ):
            fields.append("gen_uid")
            fields.append("extra_fields")
        if self.config.algorithm.adv_estimator == core_algos.AdvantageEstimator.REMAX:
            fields.append("reward_baselines")
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        curriculum_task_ids = None
        curriculum_static_weights = None
        curriculum_base_reward_scores = None
        if curriculum_enabled:
            curriculum_task_ids = [str(value) for value in to_python_list(data.pop("curriculum_task_id"))]
            curriculum_static_weights = [
                float(value) for value in to_python_list(data.pop("curriculum_weight"))
            ]
            curriculum_base_reward_scores = [
                float(value) for value in to_python_list(data.pop(curriculum_base_reward_field))
            ]

        response_mask = data["response_mask"]
        data = DataProto(batch=data.to_padded_tensor())
        data.batch["token_level_scores"] = data.batch["rm_scores"]
        data.non_tensor_batch["uid"] = np.array(data.batch.pop("uid").tolist(), dtype=object)
        if "gen_uid" in data.batch:
            data.non_tensor_batch["gen_uid"] = np.array(data.batch.pop("gen_uid").tolist(), dtype=object)
        if "extra_fields" in data.batch:
            extra_fields = data.batch.pop("extra_fields").tolist()
            agent_roles = [
                extra_field.get("agent_role") if isinstance(extra_field, dict) else None
                for extra_field in extra_fields
            ]
            if any(role is not None for role in agent_roles):
                data.non_tensor_batch["agent_role"] = np.array(agent_roles, dtype=object)

        # 1. apply kl penalty to rewards
        if self.config.algorithm.use_kl_in_reward:
            data, kl_metrics = apply_kl_penalty(
                data, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
            )
            metrics.update(kl_metrics)
        else:
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]

        # 2. Compute rollout correction: IS weights, rejection sampling, and metrics
        # Only runs in decoupled mode (computes once per batch using stable π_old)
        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        rollout_correction = (
            rollout_corr_config is not None and "rollout_log_probs" in data.batch and not bypass_recomputing_logprobs
        )
        if rollout_correction:
            data, is_metrics = compute_rollout_correction_and_add_to_batch(data, rollout_corr_config)
            metrics.update(is_metrics)

        # 3. compute advantages
        data = compute_advantage_for_multi_trajectories(
            data,
            batch_keys=batch.keys,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.config.algorithm,
        )

        # Outcome GRPO remains sequence-level. Process advantages are already
        # localized to the assistant action that caused an observed error.
        if "process_advantages" in data.batch:
            process_advantages = data.batch["process_advantages"] * data.batch["response_mask"]
            active = (process_advantages != 0) & data.batch["response_mask"].bool()
            metrics["process_reward/token_ratio"] = active.float().sum().item() / max(
                1, data.batch["response_mask"].sum().item()
            )
            metrics["process_reward/advantage_mean"] = (
                process_advantages[active].mean().item() if active.any() else 0.0
            )

        if curriculum_enabled:
            self._apply_curriculum_advantage_weights(
                data=data,
                batch_keys=batch.keys,
                task_ids=curriculum_task_ids,
                static_weights=curriculum_static_weights,
                base_reward_scores=curriculum_base_reward_scores,
                config=curriculum_config,
                metrics=metrics,
            )

        # 4. write nested advantages and returns back to TransferQueue
        fields = ["advantages", "returns"]
        if self.config.algorithm.use_kl_in_reward:
            fields.append("token_level_rewards")
        if rollout_correction:
            fields.append("response_mask")
            if "rollout_is_weights" in data.batch:
                fields.append("rollout_is_weights")

        output = {}
        for field in fields:
            output[field] = response_to_nested(data.batch[field], response_mask)
        output = TensorDict(output, batch_size=len(batch))

        extra_info = dict(batch.extra_info)
        batch = tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)
        batch.extra_info.update(extra_info)

        return batch

    def _apply_curriculum_advantage_weights(
        self,
        *,
        data: DataProto,
        batch_keys: list[str],
        task_ids: list[str],
        static_weights: list[float],
        base_reward_scores: list[float],
        config: Any,
        metrics: dict[str, float],
    ) -> None:
        """Scale gradients using static curriculum weights and base-reward EMA."""
        if (
            len(task_ids) != len(data)
            or len(static_weights) != len(data)
            or len(base_reward_scores) != len(data)
        ):
            raise ValueError(
                "curriculum metadata does not align with rollout rows: "
                f"tasks={len(task_ids)} weights={len(static_weights)} "
                f"base_rewards={len(base_reward_scores)} rows={len(data)}"
            )

        final_sessions: dict[str, tuple[int, int]] = {}
        for row_index, key in enumerate(batch_keys):
            fields = key.rsplit("_", 2)
            if len(fields) != 3:
                raise ValueError(f"unexpected rollout key for curriculum weighting: {key}")
            session_key = f"{fields[0]}_{fields[1]}"
            output_index = int(fields[2])
            if session_key not in final_sessions or final_sessions[session_key][0] < output_index:
                final_sessions[session_key] = (output_index, row_index)

        outcomes_by_task: defaultdict[str, list[float]] = defaultdict(list)
        for _, row_index in final_sessions.values():
            outcomes_by_task[task_ids[row_index]].append(float(base_reward_scores[row_index] > 0))

        ema_decay = float(config.get("ema_decay", 0.75))
        min_observations = int(config.get("min_observations", 2))
        lower_success_rate = float(config.get("lower_success_rate", 0.05))
        upper_success_rate = float(config.get("upper_success_rate", 0.85))
        inactive_weight = float(config.get("inactive_weight", 0.25))
        if not 0 <= ema_decay < 1:
            raise ValueError("algorithm.curriculum.ema_decay must be in [0, 1)")
        if not 0 <= lower_success_rate < upper_success_rate <= 1:
            raise ValueError("algorithm.curriculum success-rate thresholds are invalid")
        if inactive_weight < 0:
            raise ValueError("algorithm.curriculum.inactive_weight must be non-negative")

        dynamic_by_task: dict[str, float] = {}
        batch_success_rates: list[float] = []
        for task_id, outcomes in outcomes_by_task.items():
            current_rate = sum(outcomes) / len(outcomes)
            previous = self._curriculum_success_ema.get(task_id)
            ema = current_rate if previous is None else ema_decay * previous + (1 - ema_decay) * current_rate
            self._curriculum_success_ema[task_id] = ema
            self._curriculum_observations[task_id] += 1
            observations = self._curriculum_observations[task_id]
            inactive = observations >= min_observations and (
                ema <= lower_success_rate or ema >= upper_success_rate
            )
            dynamic_by_task[task_id] = inactive_weight if inactive else 1.0
            batch_success_rates.append(current_rate)

        static_tensor = torch.tensor(
            static_weights,
            device=data.batch["advantages"].device,
            dtype=data.batch["advantages"].dtype,
        )
        dynamic_tensor = torch.tensor(
            [dynamic_by_task[task_id] for task_id in task_ids],
            device=data.batch["advantages"].device,
            dtype=data.batch["advantages"].dtype,
        )
        effective_weights = static_tensor * dynamic_tensor
        data.batch["advantages"] = data.batch["advantages"] * effective_weights.unsqueeze(-1)
        data.batch["returns"] = data.batch["returns"] * effective_weights.unsqueeze(-1)

        task_dynamic_weights = list(dynamic_by_task.values())
        metrics.update(
            {
                "curriculum/base_success_rate_mean": float(np.mean(batch_success_rates)),
                "curriculum/static_weight_mean": float(static_tensor.mean().item()),
                "curriculum/dynamic_weight_mean": float(np.mean(task_dynamic_weights)),
                "curriculum/effective_weight_mean": float(effective_weights.mean().item()),
                "curriculum/inactive_task_fraction": float(
                    np.mean([weight < 1.0 for weight in task_dynamic_weights])
                ),
            }
        )

    def _pre_update_metric_mode(self) -> str:
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        mode = "off" if rollout_corr_config is None else rollout_corr_config.get("pre_update_metric_mode", "off")
        mode = str(mode or "off").lower()
        mode = {"none": "off", "false": "off", "0": "off", "first": "first_minibatch", "full": "full_batch"}.get(
            mode, mode
        )
        if mode not in {"off", "first_minibatch", "full_batch"}:
            raise ValueError(
                "algorithm.rollout_correction.pre_update_metric_mode must be one of "
                "'off', 'first_minibatch', or 'full_batch'."
            )
        return mode

    def _compute_pre_update_logprob_metrics(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute full-batch rollout-vs-current logprob metrics before any actor update."""
        if not self.config.actor_rollout_ref.rollout.calculate_log_probs:
            logger.warning("pre_update_metric_mode=full_batch requires rollout.calculate_log_probs=True; skipping.")
            return batch

        batch.extra_info.update(
            {
                "calculate_entropy": False,
                "compute_loss": False,
                "no_lora_adapter": False,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            }
        )
        output: KVBatchMeta = self.actor_rollout_wg.compute_log_prob(batch)
        assert len(output) == len(batch)

        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["log_probs", "response_mask", "rollout_log_probs"],
        )
        data["pre_update_log_probs"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
        total_response_lengths = data["response_mask"].offsets().diff()
        data = DataProto(
            batch=data.select("pre_update_log_probs", "rollout_log_probs", "response_mask").to_padded_tensor()
        )
        metrics.update(
            compute_compact_offpolicy_metrics(
                old_log_prob=data.batch["pre_update_log_probs"],
                rollout_log_prob=data.batch["rollout_log_probs"],
                response_mask=data.batch["response_mask"].bool(),
                prefix="actor/pre_update/",
            )
        )
        metrics.update(
            compute_detailed_offpolicy_metrics(
                old_log_prob=data.batch["pre_update_log_probs"],
                rollout_log_prob=data.batch["rollout_log_probs"],
                response_mask=data.batch["response_mask"].bool(),
                total_response_lengths=total_response_lengths,
                prefix="actor/pre_update/",
            )
        )
        return batch

    def _update_critic(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Update the critic network."""
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        extra_info = {
            "global_batch_size": ppo_mini_batch_size,
            "mini_batch_size": ppo_mini_batch_size,
            "epochs": self.config.critic.ppo_epochs,
            "seed": self.config.critic.data_loader_seed,
            "dataloader_kwargs": {"shuffle": self.config.critic.shuffle},
        }
        batch.extra_info.update(extra_info)

        output: DataProtoFuture = self.critic_wg.train_mini_batch(batch)
        output: TensorDict = output.get()
        output = rename_dict(output["metrics"], "critic/")
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        critic_metrics = reduce_metrics(output)
        metrics.update(critic_metrics)

        return batch

    def _update_actor(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Update the actor network."""
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        distillation_use_topk = (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )
        extra_info = {
            "calculate_entropy": calculate_entropy,
            "distillation_use_topk": distillation_use_topk,
            "global_batch_size": ppo_mini_batch_size,
            "mini_batch_size": ppo_mini_batch_size,
            "epochs": self.config.actor_rollout_ref.actor.ppo_epochs,
            "seed": self.config.actor_rollout_ref.actor.data_loader_seed,
            "dataloader_kwargs": {"shuffle": self.config.actor_rollout_ref.actor.shuffle},
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
            "pre_update_metric_mode": (
                "first_minibatch" if self._pre_update_metric_mode() == "first_minibatch" else "off"
            ),
        }
        batch.extra_info.update(extra_info)

        output: TensorDict = self.actor_rollout_wg.update_actor(batch)
        output = rename_dict(output["metrics"], "actor/")
        output["perf/mfu/actor"] = output.pop("actor/mfu")
        actor_metrics = reduce_metrics(output)
        metrics.update(actor_metrics)

        return batch

    def _consume_vllm_spec_decode_metrics(self) -> dict[str, float]:
        """Consume vLLM engine-level speculative decoding stats written by rollout servers."""
        stats_path = os.environ.get("VERL_VLLM_SPEC_STATS_PATH")
        if not stats_path:
            return {}

        metrics: dict[str, float] = {"rollout/spec_stats_records": 0}
        if not os.path.exists(stats_path):
            return metrics

        offset = getattr(self, "_vllm_spec_stats_file_offset", 0)
        file_size = os.path.getsize(stats_path)
        if offset > file_size:
            offset = 0

        with open(stats_path, "r", encoding="utf-8") as f:
            f.seek(offset)
            chunk = f.read()
            end_offset = f.tell()

        if chunk and not chunk.endswith("\n"):
            last_newline = chunk.rfind("\n")
            if last_newline < 0:
                self._vllm_spec_stats_file_offset = offset
                return metrics
            end_offset = offset + last_newline + 1
            chunk = chunk[: last_newline + 1]

        self._vllm_spec_stats_file_offset = end_offset
        records = []
        for line in chunk.splitlines():
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        if not records:
            return metrics

        num_drafts = sum(int(record.get("num_drafts", 0)) for record in records)
        num_draft_tokens = sum(int(record.get("num_draft_tokens", 0)) for record in records)
        num_accepted_tokens = sum(int(record.get("num_accepted_tokens", 0)) for record in records)
        accepted_tokens_per_pos = []
        for record in records:
            values = record.get("num_accepted_tokens_per_pos") or []
            if len(accepted_tokens_per_pos) < len(values):
                accepted_tokens_per_pos.extend([0] * (len(values) - len(accepted_tokens_per_pos)))
            for idx, value in enumerate(values):
                accepted_tokens_per_pos[idx] += int(value)

        metrics.update(
            {
                "rollout/spec_stats_records": len(records),
                "rollout/spec_num_drafts": num_drafts,
                "rollout/spec_num_draft_tokens": num_draft_tokens,
                "rollout/spec_num_accepted_tokens": num_accepted_tokens,
                "rollout/spec_accept_rate": num_accepted_tokens / num_draft_tokens if num_draft_tokens > 0 else 0.0,
                "rollout/spec_accept_length": 1 + num_accepted_tokens / num_drafts if num_drafts > 0 else 0.0,
            }
        )
        if num_drafts > 0:
            for idx, value in enumerate(accepted_tokens_per_pos):
                metrics[f"rollout/spec_accept_pos_{idx}"] = value / num_drafts
        return metrics

    def _compute_metrics(self, batch: KVBatchMeta, metrics, timing_raw, global_steps, epoch):
        kv_batch_extra_info = batch.extra_info

        # 1. collect necessary fields from TransferQueue for computing metrics
        non_padding_mask = np.array([not tag.get("is_padding", False) for tag in batch.tags], dtype=bool)
        fields = [
            "prompts",
            "responses",
            "response_mask",
            "values",
            "advantages",
            "returns",
            "rm_scores",
            "token_level_rewards",
            "num_turns",
            "gen_uid",
        ]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        num_turns = np.array(data.pop("num_turns").tolist())
        gen_uid = np.array(data.pop("gen_uid").tolist(), dtype=object)
        prompt_length = data["prompts"].offsets().diff()
        response_length = data["responses"].offsets().diff()
        response_items = data["responses"].unbind(0)
        response_mask_items = data["response_mask"].unbind(0)
        response_digests = [token_ids_digest(response_ids) for response_ids in response_items]
        generated_response_lengths = []
        generated_response_digests = []
        for response_ids, response_mask in zip(response_items, response_mask_items, strict=True):
            generated_response_ids = response_ids[response_mask.to(dtype=torch.bool)]
            generated_response_lengths.append(int(generated_response_ids.numel()))
            generated_response_digests.append(token_ids_digest(generated_response_ids))
        global_token_num = (prompt_length + response_length).tolist()
        real_global_token_num = (prompt_length[non_padding_mask] + response_length[non_padding_mask]).tolist()
        real_response_lengths = response_length[non_padding_mask].tolist()
        real_generated_response_lengths = [
            length for length, keep in zip(generated_response_lengths, non_padding_mask, strict=True) if keep
        ]
        real_response_digests = [
            digest for digest, keep in zip(response_digests, non_padding_mask, strict=True) if keep
        ]
        real_generated_response_digests = [
            digest for digest, keep in zip(generated_response_digests, non_padding_mask, strict=True) if keep
        ]
        real_batch_keys = [key for key, keep in zip(batch.keys, non_padding_mask, strict=True) if keep]

        extra_data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["extra_fields"],
        )
        extra_fields = to_python_list(extra_data["extra_fields"])

        # Only fetch speculative decoding stats when rollout writes them.
        spec_drafts = spec_accepts = spec_verifies = None
        mtp_config = getattr(self.config.actor_rollout_ref.model, "mtp", None)
        if mtp_config is not None and mtp_config.enable and mtp_config.enable_rollout and extra_fields is not None:
            spec_keys = ("spec_num_draft_tokens", "spec_num_accepted_tokens", "spec_num_verify_steps")
            if all(all(key in extra_field for key in spec_keys) for extra_field in extra_fields):
                spec_drafts = [extra_field["spec_num_draft_tokens"] for extra_field in extra_fields]
                spec_accepts = [extra_field["spec_num_accepted_tokens"] for extra_field in extra_fields]
                spec_verifies = [extra_field["spec_num_verify_steps"] for extra_field in extra_fields]

        if extra_fields is not None:
            prefix_cached_tokens = []
            prefix_query_tokens = []
            prefix_queries = []
            for extra_field, keep in zip(extra_fields, non_padding_mask, strict=True):
                if not keep or not isinstance(extra_field, dict):
                    continue
                cached = extra_field.get("prefix_cache_cached_tokens")
                queries = extra_field.get("prefix_cache_queries")
                query_tokens = extra_field.get("prefix_cache_query_tokens")
                if cached is None or query_tokens is None:
                    continue
                prefix_cached_tokens.append(max(0, int_or_default(cached, 0)))
                prefix_query_tokens.append(max(0, int_or_default(query_tokens, 0)))
                prefix_queries.append(max(0, int_or_default(queries, 0)))

        non_tensor_batch = {"gen_uid": gen_uid, "__num_turns__": num_turns}
        agent_roles = [
            extra_field.get("agent_role") if isinstance(extra_field, dict) else None
            for extra_field in extra_fields
        ]
        if any(role is not None for role in agent_roles):
            non_tensor_batch["agent_role"] = np.array(agent_roles, dtype=object)

        reward_extra_infos = [
            extra_field.get("reward_extra_info", {}) if isinstance(extra_field, dict) else {}
            for extra_field in extra_fields
        ]
        reward_extra_keys = sorted({key for info in reward_extra_infos for key in info.keys() if "/" in key})
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info.get(key) for info in reward_extra_infos], dtype=object)

        data = data.to_padded_tensor()
        data["token_level_scores"] = data["rm_scores"]
        if "token_level_rewards" not in data:
            data["token_level_rewards"] = data["rm_scores"]
        data["prompt_length"] = prompt_length.float()
        data["response_length"] = response_length.float()
        batch = DataProto(
            batch=data,
            non_tensor_batch=non_tensor_batch,
            meta_info={
                "global_token_num": global_token_num,
                "n": self.config.actor_rollout_ref.rollout.n,
            },
        )
        metrics_batch = batch.select_idxs(non_padding_mask) if non_padding_mask.any() else batch

        # 2. compute metrics
        metrics.update({"training/global_step": global_steps, "training/epoch": epoch})
        metrics.update(compute_data_metrics(batch=metrics_batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
        real_total_num_tokens = sum(real_global_token_num)
        metrics.update(
            {
                "padding/trajectories": int((~non_padding_mask).sum()),
                "padding/real_trajectories": int(non_padding_mask.sum()),
                "padding/total_trajectories": int(len(non_padding_mask)),
                "padding/trajectory_ratio": float((~non_padding_mask).sum() / max(len(non_padding_mask), 1)),
                "perf/real_total_num_tokens": real_total_num_tokens,
                "perf/real_throughput": real_total_num_tokens / (timing_raw["step"] * n_gpus),
            }
        )
        gradient_norm = metrics.get("actor/grad_norm", None)
        metrics.update(compute_variance_proxy_metrics(batch=metrics_batch, gradient_norm=gradient_norm))

        # 3. other auxiliary metrics
        if non_padding_mask.any():
            num_turns = num_turns[non_padding_mask]
        metrics.update(
            {
                "training/num_turns/mean": num_turns.mean(),
                "training/num_turns/max": num_turns.max(),
                "training/num_turns/min": num_turns.min(),
            }
        )

        # 4. per-request speculative-decoding aggregation (same metrics async PPO logs;
        # see compute_spec_decode_metrics in verl/trainer/ppo/ray_trainer.py).
        metrics.update(compute_spec_decode_metrics(spec_drafts, spec_accepts, spec_verifies, non_padding_mask))
        metrics.update(self._consume_vllm_spec_decode_metrics())
        if extra_fields is not None and prefix_cached_tokens:
            total_cached_tokens = sum(prefix_cached_tokens)
            total_query_tokens = sum(prefix_query_tokens)
            total_queries = sum(prefix_queries)
            metrics.update(
                {
                    "rollout/prefix_cache_hit_rate": (
                        total_cached_tokens / total_query_tokens if total_query_tokens > 0 else 0.0
                    ),
                    "rollout/prefix_cache_cached_tokens": total_cached_tokens,
                    "rollout/prefix_cache_query_tokens": total_query_tokens,
                    "rollout/prefix_cache_queries": total_queries,
                }
            )

        if extra_fields is not None:
            real_extra_fields = [
                extra_field for extra_field, keep in zip(extra_fields, non_padding_mask, strict=True) if keep
            ]
            if real_extra_fields:
                weight_step_ranges = []
                lora_applied = []
                for extra_field in real_extra_fields:
                    min_step = extra_field.get("min_global_steps", extra_field.get("global_steps"))
                    max_step = extra_field.get("max_global_steps", extra_field.get("global_steps"))
                    if min_step is not None and max_step is not None:
                        weight_step_ranges.append((int(min_step), int(max_step)))
                    if "lora_applied" in extra_field:
                        lora_applied.append(bool(extra_field["lora_applied"]))

                expected_weight_step = max(int(global_steps) - 1, 0)
                stale_weight_requests = sum(
                    min_step != expected_weight_step or max_step != expected_weight_step
                    for min_step, max_step in weight_step_ranges
                )
                metrics.update(
                    {
                        "rollout/weight_step_expected": expected_weight_step,
                        "rollout/weight_step_observed_min": (
                            min(min_step for min_step, _ in weight_step_ranges)
                            if weight_step_ranges
                            else -1
                        ),
                        "rollout/weight_step_observed_max": (
                            max(max_step for _, max_step in weight_step_ranges)
                            if weight_step_ranges
                            else -1
                        ),
                        "rollout/weight_step_missing_requests": len(real_extra_fields) - len(weight_step_ranges),
                        "rollout/weight_step_stale_requests": stale_weight_requests,
                        "rollout/lora_missing_requests": sum(not applied for applied in lora_applied),
                    }
                )
                missing_weight_requests = len(real_extra_fields) - len(weight_step_ranges)
                if missing_weight_requests and os.environ.get("VERL_ROLLOUT_WEIGHT_VERSION_STRICT", "0").lower() in {
                    "1",
                    "true",
                    "yes",
                }:
                    raise RuntimeError(
                        "Rollout did not report a model-weight version for one or more trajectories: "
                        f"missing={missing_weight_requests}/{len(real_extra_fields)}"
                    )
                if stale_weight_requests:
                    raise RuntimeError(
                        "Rollout used stale or mixed model weights: "
                        f"expected_step={expected_weight_step}, ranges={weight_step_ranges[:20]}"
                    )
                if lora_applied and not all(lora_applied):
                    raise RuntimeError(
                        "Rollout completed without the LoRA adapter on one or more trajectories: "
                        f"missing={sum(not applied for applied in lora_applied)}/{len(lora_applied)}"
                    )

                rollout_segments = [
                    int(extra_field.get("partial_rollout_segments", 1)) for extra_field in real_extra_fields
                ]
                resumed = [
                    bool(extra_field.get("partial_rollout_resumed", segments > 1))
                    for extra_field, segments in zip(real_extra_fields, rollout_segments, strict=True)
                ]
                token_count_mismatches = 0
                segment_count_mismatches = 0
                total_token_count_mismatches = 0
                token_digest_mismatches = 0
                mismatch_examples = []
                for (
                    key,
                    extra_field,
                    response_len,
                    generated_response_len,
                    response_digest,
                    generated_response_digest,
                    segments,
                    is_resumed,
                ) in zip(
                    real_batch_keys,
                    real_extra_fields,
                    real_response_lengths,
                    real_generated_response_lengths,
                    real_response_digests,
                    real_generated_response_digests,
                    rollout_segments,
                    resumed,
                    strict=True,
                ):
                    raw_counts_str = extra_field.get("partial_rollout_segment_token_counts_str", None)
                    if raw_counts_str is not None:
                        counts = [int(item) for item in str(raw_counts_str).split(",") if item.strip()]
                    else:
                        raw_counts = extra_field.get("partial_rollout_segment_token_counts", None)
                        if raw_counts is None:
                            counts = [int_or_default(generated_response_len, 0)]
                        else:
                            counts = [int_or_default(item, 0) for item in to_python_list(raw_counts)]

                    response_len_int = int_or_default(response_len, 0)
                    generated_response_len_int = int_or_default(generated_response_len, 0)
                    total_response_tokens = int_or_default(
                        extra_field.get("partial_rollout_total_response_tokens"),
                        sum(counts),
                    )
                    expected_digest = str(extra_field.get("partial_rollout_token_digest", ""))
                    count_mismatch = sum(counts) != total_response_tokens
                    total_count_mismatch = total_response_tokens != generated_response_len_int
                    digest_mismatch = bool(expected_digest) and expected_digest != generated_response_digest
                    if count_mismatch:
                        token_count_mismatches += 1
                    if total_count_mismatch:
                        total_token_count_mismatches += 1
                    if digest_mismatch:
                        token_digest_mismatches += 1
                    if len(counts) != segments:
                        segment_count_mismatches += 1
                    if (count_mismatch or total_count_mismatch or digest_mismatch or len(counts) != segments) and len(
                        mismatch_examples
                    ) < 5:
                        mismatch_examples.append(
                            {
                                "key": str(key),
                                "response_len": response_len_int,
                                "generated_response_len": generated_response_len_int,
                                "total_response_tokens": total_response_tokens,
                                "segment_counts": counts,
                                "segments": segments,
                                "resumed": is_resumed,
                                "expected_digest": expected_digest[:12],
                                "actual_digest": response_digest[:12],
                                "actual_generated_digest": generated_response_digest[:12],
                            }
                        )
                metrics.update(
                    {
                        "partial_rollout/resumed_requests": int(sum(resumed)),
                        "partial_rollout/resumed_ratio": float(sum(resumed) / len(resumed)),
                        "partial_rollout/segments_max": int(max(rollout_segments)),
                        "partial_rollout/token_count_mismatch_requests": token_count_mismatches,
                        "partial_rollout/total_token_count_mismatch_requests": total_token_count_mismatches,
                        "partial_rollout/token_digest_mismatch_requests": token_digest_mismatches,
                        "partial_rollout/segment_count_mismatch_requests": segment_count_mismatches,
                    }
                )
                if self.grouped_partial_enabled and (
                    token_count_mismatches or total_token_count_mismatches or token_digest_mismatches
                ):
                    raise RuntimeError(
                        "Partial rollout integrity check failed: "
                        f"segment_count_mismatches={token_count_mismatches}, "
                        f"total_token_count_mismatches={total_token_count_mismatches}, "
                        f"token_digest_mismatches={token_digest_mismatches}, "
                        f"examples={mismatch_examples}"
                    )

        if kv_batch_extra_info:
            for source_key, metric_key in GROUPED_PARTIAL_METRIC_ALIASES.items():
                if source_key in kv_batch_extra_info:
                    metrics[metric_key] = kv_batch_extra_info[source_key]

    def fit(self):
        if self._dump_executor._shutdown:
            self._init_dump_executor()

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(global_steps=self.global_steps)
        self._resume_rollout_generation()
        self._open_rollout_gate()

        early_stopping_config = self.config.trainer.get("early_stopping", None)
        early_stopping_enabled = bool(
            early_stopping_config and early_stopping_config.get("enabled", False)
        )
        early_stopping_metric = (
            str(early_stopping_config.get("metric")) if early_stopping_enabled else ""
        )
        early_stopping_mode = (
            str(early_stopping_config.get("mode", "max")).lower() if early_stopping_enabled else "max"
        )
        if early_stopping_mode not in {"max", "min"}:
            raise ValueError("trainer.early_stopping.mode must be 'max' or 'min'")
        best_val_score = None
        bad_validation_count = 0

        # perform validation before training
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            self.logger.log(data=val_metrics, step=self.global_steps)
            if early_stopping_enabled:
                if early_stopping_metric not in val_metrics:
                    raise KeyError(
                        f"early-stopping metric {early_stopping_metric!r} is absent from validation metrics"
                    )
                best_val_score = float(val_metrics[early_stopping_metric])
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
                return

        current_epoch = self.global_steps // len(self.train_dataloader)
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.prev_step_profile = False
        self.curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        self.next_step_profile = False

        last_val_metrics = None
        epoch = current_epoch
        while self.global_steps <= self.total_training_steps:
            for batch_dict in self.train_dataloader:
                is_last_step = self.global_steps >= self.total_training_steps
                metrics, timing_raw = {}, {}

                # 1. perform rollout and actor/critic training
                self._start_profiling()
                with marked_timer("step", timing_raw):
                    batch = self.step(batch_dict, metrics, timing_raw)

                    # 2. save checkpoint
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    # 3. update weights from trainer to rollout
                    with marked_timer("update_weights", timing_raw, color="red"):
                        self.checkpoint_manager.update_weights(global_steps=self.global_steps)
                    self._resume_rollout_generation()
                    self._open_rollout_gate()
                self._stop_profiling()

                # 4. validate
                stop_after_step = False
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)
                    if early_stopping_enabled:
                        if early_stopping_metric not in val_metrics:
                            raise KeyError(
                                f"early-stopping metric {early_stopping_metric!r} is absent from validation metrics"
                            )
                        current_val_score = float(val_metrics[early_stopping_metric])
                        min_delta = float(early_stopping_config.get("min_delta", 0.0))
                        improved = best_val_score is None or (
                            current_val_score > best_val_score + min_delta
                            if early_stopping_mode == "max"
                            else current_val_score < best_val_score - min_delta
                        )
                        if improved:
                            best_val_score = current_val_score
                            bad_validation_count = 0
                            if early_stopping_config.get("save_best", False):
                                with marked_timer("save_best_checkpoint", timing_raw, color="green"):
                                    self._save_checkpoint()
                        else:
                            bad_validation_count += 1
                        patience = int(early_stopping_config.get("patience", 3))
                        stop_after_step = bad_validation_count >= patience
                        metrics.update(
                            {
                                "early_stopping/current_score": current_val_score,
                                "early_stopping/best_score": best_val_score,
                                "early_stopping/bad_validation_count": bad_validation_count,
                                "early_stopping/triggered": float(stop_after_step),
                            }
                        )

                # 5. record metrics
                self._compute_metrics(batch, metrics, timing_raw, global_steps=self.global_steps, epoch=epoch)

                # 6. dump rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    self._log_rollout_data(batch, timing_raw, rollout_data_dir)

                # 7. cleanup transfer queue and replay buffer
                cleanup_keys = list(batch.keys)
                if self.grouped_partial_enabled:
                    prompt_keys = set()
                    if batch.extra_info:
                        prompt_keys.update(batch.extra_info.get("grouped_partial/prompt_keys", []))
                    for key, tag in zip(batch.keys, batch.tags, strict=True):
                        if tag.get("is_padding", False):
                            continue
                        parsed = ReplayBuffer._parse_output_key(key)
                        if parsed is not None:
                            prompt_keys.add(parsed[0])
                    cleanup_keys.extend(sorted(prompt_keys))
                cleanup_keys = list(dict.fromkeys(cleanup_keys))
                tq.kv_clear(keys=cleanup_keys, partition_id=batch.partition_id)
                self.replay_buffer.remove(batch.partition_id, cleanup_keys)

                self.logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step or stop_after_step:
                    if self.grouped_partial_enabled:
                        self._close_rollout_gate()
                        self.checkpoint_manager.abort_replicas()
                        if hasattr(self.async_rollout_manager, "cancel_background_tasks"):
                            self.async_rollout_manager.cancel_background_tasks()
                    self._shutdown_dump_executor()
                    if stop_after_step:
                        pprint(
                            f"Early stopping at step {self.global_steps}: "
                            f"best {early_stopping_metric}={best_val_score}"
                        )
                    else:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
            epoch += 1

        # Ensure dump executor is shut down when training loop ends without reaching is_last_step
        self._shutdown_dump_executor()

    def step(self, batch_dict: dict, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        # 1. put batch to agent loop manager
        self._open_rollout_gate()
        submit_count = len(batch_dict["raw_prompt"])
        current_prompt_keys: list[str] = []
        current_min_prompt_groups = 0
        if self.grouped_partial_enabled:
            current_min_prompt_groups = self._grouped_partial_min_prompt_groups(submit_count)

        if submit_count > 0:
            current_prompt_keys = [str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))]
            batch_dict["uid"] = np.array(current_prompt_keys, dtype=object)
            if self.config.algorithm.adv_estimator == core_algos.AdvantageEstimator.REMAX:
                rollout_n = self.config.actor_rollout_ref.rollout.n
                sampled_batch_dict = batch_dict.copy()
                sampled_batch_dict["__do_sample__"] = np.ones(len(batch_dict["raw_prompt"]), dtype=bool)
                sampled_batch_dict["__rollout_n__"] = np.full(len(batch_dict["raw_prompt"]), rollout_n, dtype=np.int64)

                baseline_batch_dict = batch_dict.copy()
                baseline_batch_dict["uid"] = np.array(
                    [f"remax_baseline_{uid}" for uid in batch_dict["uid"]], dtype=object
                )
                baseline_batch_dict["__do_sample__"] = np.zeros(len(batch_dict["raw_prompt"]), dtype=bool)
                baseline_batch_dict["__rollout_n__"] = np.ones(len(batch_dict["raw_prompt"]), dtype=np.int64)

                batch = torch.cat(
                    [tu.get_tensordict(sampled_batch_dict), tu.get_tensordict(baseline_batch_dict)], dim=0
                )
            else:
                batch = tu.get_tensordict(batch_dict)
            tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
            self.async_rollout_manager.generate_sequences(batch)

        # 2. sample batch from replay buffer
        with marked_timer("gen", timing_raw, color="red"):
            if self.grouped_partial_enabled:
                batch = self.replay_buffer.sample_ready_prompt_groups(
                    partition_id="train",
                    current_prompt_keys=current_prompt_keys,
                    current_global_steps=self.global_steps,
                    min_current_prompt_groups=current_min_prompt_groups,
                    max_off_policy_steps=self._grouped_partial_max_off_policy_steps(),
                    default_rollout_n=int(self.config.actor_rollout_ref.rollout.n),
                    ready_grace_wait_seconds=self._grouped_partial_ready_grace_wait_seconds(),
                )
                self._close_rollout_gate()
                self.checkpoint_manager.abort_replicas()
            else:
                batch = self.replay_buffer.sample(partition_id="train", global_steps=self.global_steps)
        batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

        self.checkpoint_manager.sleep_replicas()

        # 3. [OPTIONAL] compute reward score with colocated reward model
        if self.reward_loop_manager.reward_loop_worker_handles is None:
            with marked_timer("reward", timing_raw, color="yellow"):
                batch = self._compute_reward_colocate(batch)

        if self.config.algorithm.adv_estimator == core_algos.AdvantageEstimator.REMAX:
            batch = self._add_remax_reward_baselines(batch)

        # 4. balance batch across data parallel groups
        batch = self._balance_batch(batch, metrics=metrics)

        # 5. compute old_log_prob
        with marked_timer("old_log_prob", timing_raw, color="blue"):
            batch = self._compute_old_log_prob(batch, metrics=metrics)

        # 6. [OPTIONAL] compute ref_log_prob
        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, color="olive"):
                batch = self._compute_ref_log_prob(batch, metrics=metrics)

        # 7. [OPTIONAL] compute critic values
        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                batch = self._compute_values(batch, metrics=metrics)

        # 8. compute advantage and return
        with marked_timer("adv", timing_raw, color="brown"):
            batch = self._compute_advantage(batch, metrics=metrics)

        # 9. [OPTIONAL] update critic
        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                batch = self._update_critic(batch, metrics=metrics)

        # 10. update actor
        if self.config.trainer.critic_warmup <= self.global_steps:
            if self._pre_update_metric_mode() == "full_batch":
                with marked_timer("pre_update_log_prob", timing_raw, color="blue"):
                    batch = self._compute_pre_update_logprob_metrics(batch, metrics=metrics)
            with marked_timer("update_actor", timing_raw, color="red"):
                batch = self._update_actor(batch, metrics=metrics)

        return batch


@ray.remote
class TaskRunner:
    def __init__(self) -> None:
        # role => worker class
        self.role_worker_mapping = {}
        # role => resource pool
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker to mapping."""
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        role = Role.ActorRolloutRef if need_reference_policy(config) and not ref_in_actor else Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(ActorRolloutRefWorker)
        self.mapping[role] = "global_pool"

    def add_critic_worker(self, config):
        """Add critic worker to mapping."""
        if need_critic(config):
            self.role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
            self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""

        # Global resource pool is used for actor, rollout, critic, ref
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        # Add separate resource pool for reward model if enabled
        if config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward.reward_model.n_gpus_per_node] * config.reward.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
            self.mapping[Role.RewardModel] = "reward_pool"
        else:
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node
            self.mapping[Role.RewardModel] = "global_pool"

        distillation_config = config.get("distillation")
        if is_distillation_enabled(distillation_config):
            if distillation_config.n_gpus_per_node <= 0:
                raise ValueError("config.distillation.n_gpus_per_node must be greater than 0")
            if distillation_config.nnodes <= 0:
                raise ValueError("config.distillation.nnodes must be greater than 0")

            teacher_pool = [distillation_config.n_gpus_per_node] * distillation_config.nnodes
            resource_pool_spec["teacher_pool"] = teacher_pool
            self.mapping[Role.TeacherModel] = "teacher_pool"

        self.resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def run(self, config):
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # initialize transfer queue
        tq.init(config.transfer_queue)
        trainer = None
        try:
            self.add_actor_rollout_worker(config)
            self.add_critic_worker(config)
            self.init_resource_pool_mgr(config)

            trainer = PPOTrainer(
                config=config,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=self.resource_pool_manager,
            )
            trainer.init_workers()
            trainer.fit()
        finally:
            if trainer:
                trainer.replay_buffer.close()
            tq.close()


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)

    config.transfer_queue.enable = True

    # validate config
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )

    run_ppo(config, task_runner_class=TaskRunner)


if __name__ == "__main__":
    main()
