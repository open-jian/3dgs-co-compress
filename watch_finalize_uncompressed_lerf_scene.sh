#!/usr/bin/env bash
set -euo pipefail

SCENE_NAME="$1"
GPU_ID="$2"
MODEL_ROOT="/data2/jian/outputs/wacv27_experiments/clunogs_mvp/lerf_ovs/${SCENE_NAME}/uncompressed"
LOG_ROOT="/data2/jian/outputs/wacv27_experiments/logs/finalize"

mkdir -p "${LOG_ROOT}"
while [[ ! -f "${MODEL_ROOT}/.complete" ]]; do
    sleep 15
done

cd "$(dirname "${BASH_SOURCE[0]}")"
set +e
/usr/bin/time -v ./finalize_uncompressed_lerf_scene.sh "${SCENE_NAME}" "${GPU_ID}" \
    > "${LOG_ROOT}/${SCENE_NAME}_uncompressed.log" 2>&1
status=$?
set -e
printf '%s\n' "${status}" > "${LOG_ROOT}/${SCENE_NAME}_uncompressed.status"
exit "${status}"
