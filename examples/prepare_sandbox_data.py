"""Build train/test parquet files for the sandbox agent.

Two sources, mixed into one dataset:

* GSM8K questions as tool-use tasks on a scratch trial: the model may run
  Python in the sandbox, and ``return_answer`` is matched against the gold
  answer (``reward_mode=match``).
* Generated demo task directories under ``--task_root`` that exercise both
  server-side reward types: ``verify.py`` (answer check) and ``command``
  (hidden tests run against the edited workspace).

Mount ``--task_root`` into the sandbox server as SANDBOX_TASK_ROOT.

    python examples/prepare_sandbox_data.py --output_dir ~/data/sandbox_agent --task_root ~/data/sandbox_tasks
"""

import argparse
import os
import random
import re
from pathlib import Path

import datasets
import pandas as pd


def row(prompt: str, split: str, index: int, *, task_id: str = "", ground_truth: str = "", reward_mode: str):
    return {
        "data_source": "sandbox_agent",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "agent",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {"split": split, "index": index, "task_id": task_id, "reward_mode": reward_mode},
    }


def gsm8k_rows(split: str, limit: int, local_path: str | None):
    if limit <= 0:
        return []
    dataset = datasets.load_dataset(local_path or "openai/gsm8k", "main", split=split)
    rows = []
    for index, example in enumerate(dataset.select(range(min(limit, len(dataset))))):
        answer = re.search(r"#### (\-?[0-9\.\,]+)", example["answer"]).group(1).replace(",", "")
        prompt = (
            f"{example['question']}\n\nYou may use the tools to compute. Call return_answer with only the final number."
        )
        rows.append(row(prompt, split, index, ground_truth=answer, reward_mode="match"))
    return rows


def write_sum_task(task_dir: Path, rng: random.Random) -> str:
    numbers = [rng.randint(-50, 500) for _ in range(rng.randint(20, 200))]
    (task_dir / "workspace" / "data").mkdir(parents=True)
    (task_dir / "workspace" / "data" / "values.csv").write_text(
        "id,value\n" + "".join(f"{i},{n}\n" for i, n in enumerate(numbers))
    )
    threshold = rng.randint(0, 300)
    expected = sum(n for n in numbers if n > threshold)
    (task_dir / "verify.py").write_text(
        "def verify(answer):\n"
        "    try:\n"
        f"        return int(float(answer)) == {expected}\n"
        "    except ValueError:\n"
        "        return False\n"
    )
    return (
        f"What is the sum of all `value` entries greater than {threshold} in data/values.csv? Answer with the number."
    )


def write_fix_task(task_dir: Path, rng: random.Random) -> str:
    k = rng.randint(2, 9)
    (task_dir / "workspace").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "workspace" / "stats.py").write_text(
        "def moving_sum(values, k):\n"
        '    """Return the sums of every window of k consecutive values."""\n'
        "    return [sum(values[i:i + k]) for i in range(len(values) - k)]\n"
    )
    (task_dir / "tests" / "test_stats.py").write_text(
        "import sys\nsys.path.insert(0, '.')\nfrom stats import moving_sum\n\n"
        f"values = list(range({k + 5}))\n"
        f"assert moving_sum(values, {k}) == [sum(values[i:i + {k}]) for i in range(len(values) - {k} + 1)]\n"
        "assert moving_sum([1, 2], 3) == []\n"
    )
    (task_dir / "task.yaml").write_text("reward:\n  type: command\n  command: python3 .eval/tests/test_stats.py\n")
    return "stats.moving_sum drops the last window. Fix it in place, keep the signature, then call return_answer."


def demo_rows(task_root: Path, split: str, count: int, seed: int):
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        kind = "sum" if index % 2 == 0 else "fix"
        task_id = f"demo/{split}/{kind}_{index:04d}"
        task_dir = task_root / task_id
        if task_dir.exists():
            raise FileExistsError(f"{task_dir} already exists; use a fresh --task_root")
        prompt = write_sum_task(task_dir, rng) if kind == "sum" else write_fix_task(task_dir, rng)
        rows.append(row(prompt, split, index, task_id=task_id, reward_mode="sandbox"))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task_root", required=True, help="Directory mounted as SANDBOX_TASK_ROOT.")
    parser.add_argument("--gsm8k_path", default=None, help="Local copy of openai/gsm8k.")
    parser.add_argument("--train_gsm8k", type=int, default=2000)
    parser.add_argument("--test_gsm8k", type=int, default=128)
    parser.add_argument("--train_demo", type=int, default=400)
    parser.add_argument("--test_demo", type=int, default=64)
    args = parser.parse_args()

    task_root = Path(os.path.expanduser(args.task_root))
    output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, n_gsm8k, n_demo, seed in (
        ("train", args.train_gsm8k, args.train_demo, 0),
        ("test", args.test_gsm8k, args.test_demo, 1),
    ):
        rows = gsm8k_rows(split, n_gsm8k, args.gsm8k_path) + demo_rows(task_root, split, n_demo, seed)
        pd.DataFrame(rows).to_parquet(output_dir / f"{split}.parquet")
        print(f"{split}: {len(rows)} rows -> {output_dir / f'{split}.parquet'}")


if __name__ == "__main__":
    main()
