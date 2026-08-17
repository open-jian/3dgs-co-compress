#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fcgs_root="/data2/jian/WACV27_code/FCGS"
python_bin="/home/jian/miniconda3/envs/wacv27_rdo/bin/python"
codec_root="/data2/jian/outputs/wacv27_experiments/fcgs"
semantic_root="/data2/jian/outputs/wacv27_experiments/fcgs_langsplatv2"
eval_root="/data2/jian/outputs/wacv27_experiments/fcgs_langsplatv2_eval/lerf_ovs/aggressive"
data_root="/data2/jian/data/lerf_ovs"
gpu_id="${FCGS_VALIDATE_GPU:-2}"
lock_file="/data2/jian/outputs/wacv27_experiments/logs/fcgs_curve_finalize/gpu_${gpu_id}.lock"
export PATH="/data2/jian/WACV27_code/mpeg-pcc-tmc13/build/tmc3:${PATH}"

validate_scene() {
    local scene="$1"
    local semantic_dir="${semantic_root}/lerf_ovs/${scene}/aggressive"
    local codec_dir="${codec_root}/lerf_ovs/${scene}/aggressive"
    while [[ ! -f "${semantic_dir}/.deployment_eval.complete" ]]; do
        sleep 20
    done
    if [[ -f "${codec_dir}/.rgb_validation.complete" ]]; then
        return 0
    fi
    (
        flock -x 9
        cd "${fcgs_root}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" /usr/bin/time -v "${python_bin}" \
            decode_single_scene_validate.py \
            --lmd 0.0016 \
            --bit_path_from "${codec_dir}/bitstreams" \
            --ply_path_to "${codec_dir}/decoded/point_cloud.ply" \
            --source_path "${data_root}/${scene}" \
            > "${codec_dir}/logs/validate.log" 2>&1
        touch "${codec_dir}/.rgb_validation.complete"
    ) 9>"${lock_file}"
}

for scene in ramen figurines teatime waldo_kitchen; do
    validate_scene "${scene}" &
done
wait

cd "${repo_root}"
python3 fcgs_table_summary.py \
    --codec-root "${codec_root}" \
    --semantic-root "${semantic_root}" \
    --eval-root "${eval_root}" \
    --operating-point aggressive \
    --output "${semantic_root}/lerf_ovs/fcgs_lerf_table_summary.aggressive.json"
