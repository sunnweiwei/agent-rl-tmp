#!/usr/bin/env python3
"""Print compact RL training diagnostics from a Weights & Biases run."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from typing import Any

import wandb


METRICS = [
    "_step",
    "training/global_step",
    "val-core/sandbox_agent/reward/mean@4",
    "val-core/sandbox_agent/reward/best@4/mean",
    "actor/lr",
    "actor/pre_update/kl",
    "actor/rollout_corr/kl",
    "actor/ppo_kl",
    "actor/pg_clipfrac",
    "actor/pg_clipfrac_lower",
    "actor/entropy_loss",
    "actor/grad_norm",
    "aa/agent/base_reward_score",
    "aa/agent/reward",
    "aa/agent/process_reward_sum",
    "aa/agent/parse_errors",
    "aa/agent/tool_errors",
    "aa/agent/tool_timeouts",
    "aa/agent/terminated_context_full",
    "response_length/mean",
    "response_length/max",
    "response_length/clip_ratio",
    "process_reward/token_ratio",
    "rollout/prefix_cache_hit_rate",
    "rollout/lora_missing_requests",
    "rollout/weight_step_stale_requests",
    "timing_s/gen",
    "timing_s/update_actor",
    "timing_s/step",
]


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def compact(row: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    return {key: row[key] for key in keys if finite_number(row.get(key))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="W&B path such as entity/project/run_id")
    args = parser.parse_args()

    run = wandb.Api().run(args.run)
    rows = list(run.scan_history(keys=METRICS, page_size=1000))
    train_rows = [row for row in rows if finite_number(row.get("training/global_step"))]
    val_rows = [
        row
        for row in rows
        if finite_number(row.get("val-core/sandbox_agent/reward/mean@4"))
    ]

    val_keys = [
        "_step",
        "training/global_step",
        "val-core/sandbox_agent/reward/mean@4",
        "val-core/sandbox_agent/reward/best@4/mean",
    ]
    train_keys = [key for key in METRICS if key not in val_keys[2:]]

    print(json.dumps({
        "name": run.name,
        "id": run.id,
        "state": run.state,
        "last_history_step": run.lastHistoryStep,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
    }, sort_keys=True))
    print("VALIDATION")
    for row in val_rows:
        print(json.dumps(compact(row, val_keys), sort_keys=True))
    print("TRAIN_EVERY_5")
    for row in train_rows:
        step = row.get("training/global_step")
        if int(step) == 1 or int(step) % 5 == 0:
            print(json.dumps(compact(row, train_keys), sort_keys=True))


if __name__ == "__main__":
    main()
