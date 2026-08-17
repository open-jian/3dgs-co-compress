#!/usr/bin/env bash
set -euo pipefail

SCENE_NAME="$1"
OPERATING_POINT="$2"
GPU_ID="${3:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/jian/miniconda3/envs/langsplat_v2/bin/python"
DATA_ROOT="/data2/jian/data/lerf_ovs"
PLY="/data2/jian/outputs/wacv27_experiments/fcgs/lerf_ovs/${SCENE_NAME}/${OPERATING_POINT}/decoded/point_cloud.ply"
OUTPUT_ROOT="/data2/jian/outputs/wacv27_experiments/fcgs_imported_checkpoints/lerf_ovs/${SCENE_NAME}/${OPERATING_POINT}"
CHECKPOINT="${OUTPUT_ROOT}/chkpnt30000.pth"

if [[ ! -f "${PLY}" ]]; then
    printf 'missing decoded FCGS PLY: %s\n' "${PLY}" >&2
    exit 2
fi
mkdir -p "${OUTPUT_ROOT}"
if [[ ! -f "${CHECKPOINT}" ]]; then
    cd "${REPO_ROOT}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" import_rgb_ply_checkpoint.py \
        --ply "${PLY}" \
        --source "${DATA_ROOT}/${SCENE_NAME}" \
        --output "${CHECKPOINT}" \
        --eval
fi
touch "${OUTPUT_ROOT}/.complete"
