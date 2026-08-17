#!/usr/bin/env bash
set -euo pipefail

SCENE_NAME="$1"
GPU_ID="$2"
PORT_BASE="$3"

MEDIUM_ROOT="/data2/jian/outputs/wacv27_experiments/fcgs_langsplatv2/lerf_ovs/${SCENE_NAME}"
LOG_ROOT="/data2/jian/outputs/wacv27_experiments/logs/fcgs_curve_remote"
mkdir -p "${LOG_ROOT}"

while [[ ! -f "${MEDIUM_ROOT}/.complete" ]]; do
    sleep 15
done

cd "$(dirname "${BASH_SOURCE[0]}")"
final_status=0
offset=0
for operating_point in light aggressive; do
    log_file="${LOG_ROOT}/${SCENE_NAME}_${operating_point}.log"
    status_file="${LOG_ROOT}/${SCENE_NAME}_${operating_point}.status"
    set +e
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    /usr/bin/time -v ./run_fcgs_langsplatv2_scene.sh \
        "${SCENE_NAME}" 1000 "$(( PORT_BASE + offset ))" "${operating_point}" \
        > "${log_file}" 2>&1
    status=$?
    set -e
    printf '%s\n' "${status}" > "${status_file}"
    if [[ "${status}" -ne 0 ]]; then
        final_status="${status}"
    fi
    # Scene base ports are spaced by 100.  Put the second operating point in
    # a disjoint range so that one fast scene cannot collide with the next
    # scene while their jobs overlap.
    offset=800
done
exit "${final_status}"
