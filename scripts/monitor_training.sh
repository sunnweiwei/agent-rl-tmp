#!/usr/bin/env bash
# Monitor a long-running training container and preserve crash diagnostics.
#   OUTPUT_DIR=<run output dir> bash scripts/monitor_training.sh   (or CONTAINER_NAME=...)
set -uo pipefail

CONTAINER_FILE=${CONTAINER_FILE:-${OUTPUT_DIR:-.}/container_name}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-300}
STALL_SECONDS=${STALL_SECONDS:-5400}
KL_WARN=${KL_WARN:-0.01}
VAL_DROP_WARN=${VAL_DROP_WARN:-0.15}
MONITOR_ROOT=${MONITOR_ROOT:-${OUTPUT_DIR:-.}/monitoring}
# Validation metric is val-core/<data_source>/reward/mean@<VAL_N>.
VAL_METRIC=${VAL_METRIC:-val-core/sandbox_agent/reward/mean@4}

container=${CONTAINER_NAME:-$(cat "${CONTAINER_FILE}" 2>/dev/null || true)}
if [ -z "${container}" ]; then
  echo "[monitor] no container found in ${CONTAINER_FILE}" >&2
  exit 2
fi

run_key=$(printf '%s' "${container}" | sha256sum | cut -c1-16)
run_dir="${MONITOR_ROOT}/${run_key}"
mkdir -p "${run_dir}"
status_file="${run_dir}/status.env"
history_file="${run_dir}/history.tsv"
alert_file="${run_dir}/ALERT"
started_epoch=$(date +%s)
last_step=-1
last_progress_epoch=${started_epoch}
baseline_val=""

printf 'timestamp\tstate\tstep\tval_reward\tkl\ttrain_reward\tresponse_length\tprefix_hit\tstale_weights\terrors\txids\tecc_uncorrected\n' > "${history_file}"
rm -f "${alert_file}"

number_gt() {
  awk -v left="${1:-0}" -v right="${2:-0}" 'BEGIN { exit !(left > right) }'
}

number_lt() {
  awk -v left="${1:-0}" -v right="${2:-0}" 'BEGIN { exit !(left < right) }'
}

snapshot_failure() {
  local stamp=$1
  docker logs --tail 2000 "${container}" > "${run_dir}/failure-${stamp}.log" 2>&1 || true
  sudo journalctl -k --since "@${started_epoch}" --no-pager \
    > "${run_dir}/kernel-${stamp}.log" 2>&1 || true
  nvidia-smi -q > "${run_dir}/nvidia-smi-${stamp}.log" 2>&1 || true
}

while true; do
  now_epoch=$(date +%s)
  timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  state=$(docker inspect -f '{{.State.Status}}' "${container}" 2>/dev/null || echo missing)
  exit_code=$(docker inspect -f '{{.State.ExitCode}}' "${container}" 2>/dev/null || echo -1)
  oom=$(docker inspect -f '{{.State.OOMKilled}}' "${container}" 2>/dev/null || echo unknown)

  full_log=$(mktemp)
  recent_log=$(mktemp)
  docker logs "${container}" > "${full_log}" 2>&1 || true
  docker logs --since "$((INTERVAL_SECONDS + 60))s" "${container}" > "${recent_log}" 2>&1 || true

  step=$(grep -oE 'training/global_step:[0-9]+' "${full_log}" | tail -n1 | cut -d: -f2)
  step=${step:-0}
  val_reward=$(grep -oE "${VAL_METRIC}"':np.float64\([-+0-9.eE]+\)' "${full_log}" \
    | tail -n1 | sed -E 's/.*\(([-+0-9.eE]+)\)/\1/')
  kl=$(grep -oE 'actor/pre_update/kl:np.float64\([-+0-9.eE]+\)' "${full_log}" \
    | tail -n1 | sed -E 's/.*\(([-+0-9.eE]+)\)/\1/')
  train_reward=$(grep -oE 'aa/agent/reward:[-+0-9.eE]+' "${full_log}" \
    | tail -n1 | cut -d: -f2)
  response_length=$(grep -oE 'response_length/mean:[-+0-9.eE]+' "${full_log}" \
    | tail -n1 | cut -d: -f2)
  prefix_hit=$(grep 'Engine 000:' "${full_log}" | tail -n1 \
    | sed -nE 's/.*Prefix cache hit rate: ([0-9.]+)%.*/\1/p')
  stale_weights=$(grep -oE 'rollout/weight_step_stale_requests:[-+0-9.eE]+' "${full_log}" \
    | tail -n1 | cut -d: -f2)
  stale_weights=${stale_weights:-0}

  errors=$(grep -Ec 'CUDA error|illegal memory|EngineCore encountered|OutOfMemory|RayTaskError|Traceback' "${recent_log}" || true)
  xids=$(sudo journalctl -k --since "@${started_epoch}" --no-pager 2>/dev/null \
    | grep -c 'NVRM: Xid' || true)
  ecc_uncorrected=$(nvidia-smi \
    --query-gpu=ecc.errors.uncorrected.volatile.device_memory \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk '{ total += $1 } END { print total + 0 }')

  if [ "${step}" -gt "${last_step}" ]; then
    last_step=${step}
    last_progress_epoch=${now_epoch}
  fi
  if [ -z "${baseline_val}" ] && [ -n "${val_reward}" ]; then
    baseline_val=${val_reward}
  fi

  alerts=()
  [ "${state}" = running ] || alerts+=("container_${state}_exit_${exit_code}_oom_${oom}")
  [ "${errors}" -eq 0 ] || alerts+=("runtime_errors_${errors}")
  [ "${xids}" -eq 0 ] || alerts+=("gpu_xid_${xids}")
  [ "${ecc_uncorrected}" -eq 0 ] || alerts+=("gpu_ecc_uncorrected_${ecc_uncorrected}")
  if number_gt "${stale_weights}" 0; then
    alerts+=("stale_rollout_weights_${stale_weights}")
  fi
  if [ -n "${kl}" ] && number_gt "${kl}" "${KL_WARN}"; then
    alerts+=("kl_high_${kl}")
  fi
  if [ -n "${baseline_val}" ] && [ -n "${val_reward}" ]; then
    val_floor=$(awk -v base="${baseline_val}" -v drop="${VAL_DROP_WARN}" 'BEGIN { print base - drop }')
    if number_lt "${val_reward}" "${val_floor}"; then
      alerts+=("val_reward_drop_${val_reward}_baseline_${baseline_val}")
    fi
  fi
  if [ "${step}" -gt 0 ] && [ -n "${prefix_hit}" ] && number_lt "${prefix_hit}" 85; then
    alerts+=("prefix_hit_low_${prefix_hit}")
  fi
  if [ $((now_epoch - last_progress_epoch)) -gt "${STALL_SECONDS}" ]; then
    alerts+=("no_step_progress_$((now_epoch - last_progress_epoch))s")
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${timestamp}" "${state}" "${step}" "${val_reward:-}" "${kl:-}" \
    "${train_reward:-}" "${response_length:-}" "${prefix_hit:-}" \
    "${stale_weights}" "${errors}" "${xids}" "${ecc_uncorrected}" >> "${history_file}"

  status_tmp="${status_file}.tmp"
  {
    printf 'timestamp=%q\n' "${timestamp}"
    printf 'container=%q\n' "${container}"
    printf 'state=%q\n' "${state}"
    printf 'exit_code=%q\n' "${exit_code}"
    printf 'oom_killed=%q\n' "${oom}"
    printf 'step=%q\n' "${step}"
    printf 'val_reward=%q\n' "${val_reward:-}"
    printf 'baseline_val_reward=%q\n' "${baseline_val:-}"
    printf 'kl=%q\n' "${kl:-}"
    printf 'train_reward=%q\n' "${train_reward:-}"
    printf 'response_length=%q\n' "${response_length:-}"
    printf 'prefix_cache_hit_percent=%q\n' "${prefix_hit:-}"
    printf 'stale_rollout_weights=%q\n' "${stale_weights}"
    printf 'recent_errors=%q\n' "${errors}"
    printf 'xids_since_monitor_start=%q\n' "${xids}"
    printf 'ecc_uncorrected=%q\n' "${ecc_uncorrected}"
    printf 'alerts=%q\n' "${alerts[*]:-}"
  } > "${status_tmp}"
  mv "${status_tmp}" "${status_file}"

  rm -f "${full_log}" "${recent_log}"
  if [ "${#alerts[@]}" -gt 0 ]; then
    printf '%s %s\n' "${timestamp}" "${alerts[*]}" | tee -a "${alert_file}" >&2
  fi
  if [ "${state}" != running ]; then
    snapshot_failure "$(date -u +%Y%m%d-%H%M%S)"
    exit 1
  fi
  sleep "${INTERVAL_SECONDS}"
done
