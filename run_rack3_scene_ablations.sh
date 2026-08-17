#!/usr/bin/env bash
set -uo pipefail

# Keep one scene on one rack3 GPU: stop-gradient first, then sequential.
# Both child jobs keep independent log/status/provenance records.

SCENE_NAME="${1:?SCENE is required}"
GPU_INDEX="${2:?GPU is required}"
PORT_BASE="${3:?PORT_BASE is required}"
SOURCE_CHECKPOINT="${4:?SOURCE_CKPT is required}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/data2/jian/outputs/wacv27_experiments}"
PIPELINE_STATUS_ROOT="${EXPERIMENT_ROOT}/logs/core_ablations/pipelines"
mkdir -p "${PIPELINE_STATUS_ROOT}"

"${REPO_ROOT}/run_rack3_core_ablation_job.sh" \
    stop_grad "${SCENE_NAME}" "${GPU_INDEX}" "${PORT_BASE}" "${SOURCE_CHECKPOINT}"
STOP_STATUS=$?

"${REPO_ROOT}/run_rack3_core_ablation_job.sh" \
    sequential "${SCENE_NAME}" "${GPU_INDEX}" "$(( PORT_BASE + 100 ))" "${SOURCE_CHECKPOINT}"
SEQUENTIAL_STATUS=$?

printf 'stop_grad=%s\nsequential=%s\n' \
    "${STOP_STATUS}" "${SEQUENTIAL_STATUS}" \
    > "${PIPELINE_STATUS_ROOT}/${SCENE_NAME}.status"

if (( STOP_STATUS != 0 || SEQUENTIAL_STATUS != 0 )); then
    exit 1
fi
