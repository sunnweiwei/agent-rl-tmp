#!/usr/bin/env bash
# Run the no-MTP/MTP prefix-cache correctness comparison on one B200.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
IMAGE=${IMAGE:-sunnweiwei/verl-qwen35:b200-verl0.8.0-vllm-main-c621af169-fla051-triton-gdn-20260716}
MODEL_PATH=${MODEL_PATH:-/mnt/workspace/models/Qwen3.6-27B}
GPU=${GPU:-0}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/workspace/outputs/b200_mtp_prefix_cache_correctness_$(date +%Y%m%d-%H%M%S)}
VARIANTS=${VARIANTS:-"no_mtp mtp1"}

mkdir -p "${OUTPUT_DIR}"

run_variant() {
    local variant=$1
    shift
    docker run --rm \
        --device "nvidia.com/gpu=${GPU}" \
        --ipc host \
        --network host \
        --shm-size 64g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -e CUDA_VISIBLE_DEVICES="${GPU}" \
        -e NVIDIA_VISIBLE_DEVICES="${GPU}" \
        -e FLA_TILELANG=0 \
        -e TOKENIZERS_PARALLELISM=false \
        -v "${REPO_DIR}:/workspace/agent-rl:ro" \
        -v "${MODEL_PATH}:${MODEL_PATH}:ro" \
        -v "${OUTPUT_DIR}:/workspace/results" \
        -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
        -v "${HOME}/.cache/vllm:/root/.cache/vllm" \
        -v "${HOME}/.cache/torchinductor:/root/.cache/torchinductor" \
        -v "${HOME}/.triton:/root/.triton" \
        -w /workspace/agent-rl \
        "${IMAGE}" \
        python3 scripts/diagnostics/b200_mtp_prefix_cache_correctness.py \
            --model "${MODEL_PATH}" \
            --variant "${variant}" \
            --output-json "/workspace/results/${variant}.json" \
            "$@" \
        2>&1 | tee "${OUTPUT_DIR}/${variant}.log"
}

for variant in ${VARIANTS}; do
    case "${variant}" in
        no_mtp) run_variant no_mtp ;;
        mtp1) run_variant mtp1 --reference-json /workspace/results/no_mtp.json ;;
        *) echo "Unknown variant: ${variant}" >&2; exit 2 ;;
    esac
done

python3 - "${OUTPUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = [json.loads((root / f"{name}.json").read_text()) for name in ("no_mtp", "mtp1")]
for row in rows:
    print(
        row["variant"],
        f"passed={row['passed']}",
        f"cold={row['cold_seconds']:.3f}s",
        f"warm={row['warm_seconds']:.3f}s",
        f"prefix_hit={row['warm_prefix_cache_hit_rate']:.3%}",
        f"acceptance={row['spec_acceptance_rate']}",
    )
PY

echo "Results: ${OUTPUT_DIR}"
