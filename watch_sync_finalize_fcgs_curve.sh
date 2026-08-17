#!/usr/bin/env bash
set -euo pipefail

remote_host="${FCGS_REMOTE_HOST:-r3}"
remote_semantic_root="/data2/jian/outputs/wacv27_experiments/fcgs_langsplatv2/lerf_ovs"
local_semantic_root="/data2/jian/outputs/wacv27_experiments/fcgs_langsplatv2/lerf_ovs"
remote_log_root="/data2/jian/outputs/wacv27_experiments/logs/fcgs_curve_remote"
local_log_root="/data2/jian/outputs/wacv27_experiments/logs/fcgs_curve_remote"
watch_log_root="/data2/jian/outputs/wacv27_experiments/logs/fcgs_curve_finalize"
gpu_id="${FCGS_FINALIZE_GPU:-2}"
lock_file="${watch_log_root}/gpu_${gpu_id}.lock"

mkdir -p "${local_semantic_root}" "${local_log_root}" "${watch_log_root}"
cd "$(dirname "${BASH_SOURCE[0]}")"

sync_and_finalize() {
    local scene="$1"
    local operating_point="$2"
    local remote_dir="${remote_semantic_root}/${scene}/${operating_point}"
    local local_dir="${local_semantic_root}/${scene}/${operating_point}"
    local log_file="${watch_log_root}/${scene}_${operating_point}.log"

    if [[ -f "${local_dir}/.deployment_eval.complete" ]]; then
        return 0
    fi
    while ! ssh -o BatchMode=yes "${remote_host}" "test -f '${remote_dir}/.complete'"; do
        sleep 20
    done

    mkdir -p "${local_dir}"
    rsync -a --partial --exclude='.complete' \
        "${remote_host}:${remote_dir}/" "${local_dir}/" >>"${log_file}" 2>&1
    for suffix in log status; do
        rsync -a --partial \
            "${remote_host}:${remote_log_root}/${scene}_${operating_point}.${suffix}" \
            "${local_log_root}/" >>"${log_file}" 2>&1 || true
    done
    touch "${local_dir}/.complete" "${local_dir}/.synced_from_rack3"

    (
        flock -x 9
        ./finalize_fcgs_langsplatv2_scene.sh \
            "${scene}" "${gpu_id}" 1000 "${operating_point}" >>"${log_file}" 2>&1
    ) 9>"${lock_file}"
}

for scene in ramen figurines teatime waldo_kitchen; do
    for operating_point in light aggressive; do
        sync_and_finalize "${scene}" "${operating_point}" &
    done
done
wait
