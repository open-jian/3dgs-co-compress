#!/usr/bin/env bash
set -euo pipefail

SCENE_NAME="$1"
GPU_ID="${2:-0}"
ITERATION="${3:-1000}"
OPERATING_POINT="${4:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/jian/miniconda3/envs/langsplat_v2/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data2/jian/data/lerf_ovs}"
CODEC_ROOT="${CODEC_ROOT:-/data2/jian/outputs/wacv27_experiments/fcgs/lerf_ovs}"
RGB_IMPORT_ROOT="${RGB_IMPORT_ROOT:-/data2/jian/outputs/wacv27_experiments/fcgs_imported_checkpoints/lerf_ovs}"
SEMANTIC_ROOT="${SEMANTIC_ROOT:-/data2/jian/outputs/wacv27_experiments/fcgs_langsplatv2/lerf_ovs}"
EVAL_ROOT="${EVAL_ROOT:-/data2/jian/outputs/wacv27_experiments/fcgs_langsplatv2_eval/lerf_ovs}"

SCENE_DIR="${SEMANTIC_ROOT}/${SCENE_NAME}"
if [[ -n "${OPERATING_POINT}" ]]; then
    SCENE_DIR="${SCENE_DIR}/${OPERATING_POINT}"
    RGB_IMPORT_ROOT="${RGB_IMPORT_ROOT}/${SCENE_NAME}/${OPERATING_POINT}"
    CODEC_ROOT="${CODEC_ROOT}/${SCENE_NAME}/${OPERATING_POINT}"
    EVAL_ROOT="${EVAL_ROOT}/${OPERATING_POINT}"
else
    RGB_IMPORT_ROOT="${RGB_IMPORT_ROOT}/${SCENE_NAME}"
    CODEC_ROOT="${CODEC_ROOT}/${SCENE_NAME}"
fi
JOINT_CHECKPOINT="${SCENE_DIR}/chkpnt${ITERATION}.pth"
RGB_CHECKPOINT="${RGB_IMPORT_ROOT}/chkpnt30000.pth"
DECODED_PLY="${CODEC_ROOT}/decoded/point_cloud.ply"
SIDECAR="${SCENE_DIR}/semantic.sidecar.pt"

for required in "${JOINT_CHECKPOINT}" "${RGB_CHECKPOINT}" "${DECODED_PLY}"; do
    if [[ ! -f "${required}" ]]; then
        echo "missing required input: ${required}" >&2
        exit 2
    fi
done

mkdir -p "${SCENE_DIR}" "${EVAL_ROOT}"
cd "${REPO_ROOT}"

if [[ ! -f "${SIDECAR}" || ! -f "${SCENE_DIR}/semantic_sidecar_reload.json" ]]; then
    "${PYTHON_BIN}" semantic_sidecar.py export \
        --checkpoint "${JOINT_CHECKPOINT}" \
        --geometry-checkpoint "${RGB_CHECKPOINT}" \
        --output "${SIDECAR}" \
        --topk 4 \
        --force \
        > "${SCENE_DIR}/semantic_sidecar_export.json"

    "${PYTHON_BIN}" semantic_sidecar.py validate \
        --sidecar "${SIDECAR}" \
        --geometry-ply "${DECODED_PLY}" \
        --source-checkpoint "${JOINT_CHECKPOINT}" \
        > "${SCENE_DIR}/semantic_sidecar_reload.json"
fi

# The geometry input is a decoder output derived only from the counted FCGS
# scene bitstream and shared FCGS checkpoint; no dense RGB checkpoint is read
# by the evaluation process.
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_lerf.py \
    -s "${DATA_ROOT}/${SCENE_NAME}" \
    -m "${SCENE_DIR}" \
    --dataset_name "${SCENE_NAME}" \
    --index 0 \
    --geometry_ply "${DECODED_PLY}" \
    --semantic_sidecar "${SIDECAR}" \
    --ckpt_root_path "${SEMANTIC_ROOT}" \
    --output_dir "${EVAL_ROOT}" \
    --mask_thresh 0.4 \
    --json_folder "${DATA_ROOT}/label" \
    --checkpoint "${ITERATION}" \
    --include_feature \
    --topk 4 \
    --quick_render \
    > "${SCENE_DIR}/lerf_eval.log" 2>&1

touch "${SCENE_DIR}/.deployment_eval.complete"
