#!/usr/bin/env bash
set -uo pipefail

# Reserve ramen's GPU after its already-running stop-gradient job, run the
# required teatime full-joint job, then let the original ramen scene wrapper
# continue with its queued sequential job.

RAMEN_WRAPPER_PID="${1:?ramen wrapper PID is required}"
GPU_INDEX="${2:-4}"
TEATIME_SOURCE="${3:?teatime source checkpoint is required}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/data2/jian/outputs/wacv27_experiments}"
RAMEN_STOP_STATUS="${EXPERIMENT_ROOT}/logs/core_ablations/stop_grad/ramen.status"
PRIORITY_STATUS="${EXPERIMENT_ROOT}/logs/core_ablations/pipelines/teatime_priority.status"
PRIORITY_MARKER="${EXPERIMENT_ROOT}/logs/core_ablations/pipelines/ramen_waits_for_teatime_full"
mkdir -p "$(dirname "${PRIORITY_STATUS}")"
touch "${PRIORITY_MARKER}"

while true; do
    CURRENT_STATUS="$(cat "${RAMEN_STOP_STATUS}" 2>/dev/null || true)"
    if [[ "${CURRENT_STATUS}" =~ ^[0-9]+$ ]]; then
        break
    fi
    sleep 20
done

"${REPO_ROOT}/run_rack3_core_ablation_job.sh" \
    full_joint teatime "${GPU_INDEX}" 64604 "${TEATIME_SOURCE}"
FULL_STATUS=$?

RAMEN_SEQUENTIAL_STATUS="${EXPERIMENT_ROOT}/logs/core_ablations/sequential/ramen.status"
while true; do
    SEQUENTIAL_STATUS="$(cat "${RAMEN_SEQUENTIAL_STATUS}" 2>/dev/null || true)"
    if [[ "${SEQUENTIAL_STATUS}" =~ ^[0-9]+$ ]]; then
        break
    fi
    if ! kill -0 "${RAMEN_WRAPPER_PID}" 2>/dev/null; then
        SEQUENTIAL_STATUS=99
        break
    fi
    sleep 20
done

printf 'ramen_stop_grad=%s\nteatime_full_joint=%s\nramen_sequential=%s\n' \
    "${CURRENT_STATUS}" "${FULL_STATUS}" "${SEQUENTIAL_STATUS}" \
    > "${PRIORITY_STATUS}"

if (( CURRENT_STATUS != 0 || FULL_STATUS != 0 || SEQUENTIAL_STATUS != 0 )); then
    exit 1
fi
