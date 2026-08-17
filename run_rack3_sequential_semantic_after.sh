#!/usr/bin/env bash
set -euo pipefail

# Queue one semantic-guided sequential cell behind the existing RGB-guided
# sequential cell on the same GPU.

SCENE_NAME="${1:?SCENE is required}"
GPU_INDEX="${2:?GPU is required}"
PORT="${3:?PORT is required}"
SOURCE_CHECKPOINT="${4:?SOURCE_CKPT is required}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/data2/jian/outputs/wacv27_experiments}"
RGB_SEQ_STATUS="${EXPERIMENT_ROOT}/logs/core_ablations/sequential/${SCENE_NAME}.status"

while true; do
    STATUS="$(cat "${RGB_SEQ_STATUS}" 2>/dev/null || true)"
    if [[ "${STATUS}" =~ ^[0-9]+$ ]]; then
        break
    fi
    sleep 20
done
if (( STATUS != 0 )); then
    echo "RGB-guided sequential dependency failed with status ${STATUS}" >&2
    exit "${STATUS}"
fi

exec "${REPO_ROOT}/run_rack3_core_ablation_job.sh" \
    sequential_semantic "${SCENE_NAME}" "${GPU_INDEX}" "${PORT}" "${SOURCE_CHECKPOINT}"
