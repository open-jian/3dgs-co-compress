#!/usr/bin/env bash
set -euo pipefail

SCENE_NAME="$1"
GPU_ID="${2:-0}"
ITERATION="${3:-1000}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/jian/miniconda3/envs/langsplat_v2/bin/python"
DATA_ROOT="/data2/jian/data/lerf_ovs"
MODEL_ROOT="/data2/jian/outputs/wacv27_experiments/clunogs_mvp/lerf_ovs/${SCENE_NAME}/uncompressed"
CHECKPOINT="${MODEL_ROOT}/chkpnt${ITERATION}.pth"
ARTIFACT_ROOT="${MODEL_ROOT}/deploy"
ARTIFACT="${ARTIFACT_ROOT}/${SCENE_NAME}.uncompressed.pth"
EVAL_ROOT="${MODEL_ROOT}/evaluation"

if [[ ! -f "${CHECKPOINT}" ]]; then
    printf 'missing uncompressed checkpoint for %s\n' "${SCENE_NAME}" >&2
    exit 2
fi

mkdir -p "${ARTIFACT_ROOT}" "${EVAL_ROOT}"
cp "${MODEL_ROOT}/cfg_args" "${ARTIFACT_ROOT}/cfg_args"
cd "${REPO_ROOT}"
if [[ ! -f "${ARTIFACT}" ]]; then
    "${PYTHON_BIN}" compact_artifact.py export \
        --checkpoint "${CHECKPOINT}" \
        --output "${ARTIFACT}" \
        --topk 4
fi
"${PYTHON_BIN}" compact_artifact.py validate --artifact "${ARTIFACT}" \
    > "${ARTIFACT_ROOT}/validation.json"

if [[ ! -f "${EVAL_ROOT}/semantic/${SCENE_NAME}_0/metrics_lerf.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_lerf.py \
        -s "${DATA_ROOT}/${SCENE_NAME}" \
        -m "${ARTIFACT_ROOT}" \
        --dataset_name "${SCENE_NAME}" \
        --index 0 \
        --compact_artifact "${ARTIFACT}" \
        --ckpt_root_path "${ARTIFACT_ROOT}" \
        --output_dir "${EVAL_ROOT}/semantic" \
        --mask_thresh 0.4 \
        --json_folder "${DATA_ROOT}/label" \
        --checkpoint "${ITERATION}" \
        --include_feature \
        --topk 4 \
        --quick_render
fi

if [[ ! -f "${EVAL_ROOT}/rgb.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_compact_rgb.py \
        -s "${DATA_ROOT}/${SCENE_NAME}" \
        -m "${ARTIFACT_ROOT}" \
        --eval \
        --compact_artifact "${ARTIFACT}" \
        --output "${EVAL_ROOT}/rgb.json"
fi

touch "${MODEL_ROOT}/.finalized"
