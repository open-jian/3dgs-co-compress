#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(dirname "${REPO_ROOT}")"
WORKSPACE_ROOT="$(dirname "${CODE_ROOT}")"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_ROOT}/Output}"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE_ROOT}/Data}"

DATASET_NAME=$1
INDEX=${2:-0}
CHECKPOINT=${3:-10000}
TOPK=4
MODEL_DIR="${OUTPUT_ROOT}/langsplatv2_cluno/joint/mip_nerf360/${DATASET_NAME}_${INDEX}"
JOINT_CHECKPOINT="${JOINT_CHECKPOINT:-${MODEL_DIR}/chkpnt${CHECKPOINT}.pth}"

cd "${REPO_ROOT}"
python eval_mip_nerf360.py \
    -s "${DATA_ROOT}/Mip-NeRF360/${DATASET_NAME}" \
    -m "${MODEL_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --index "${INDEX}" \
    --joint_checkpoint "${JOINT_CHECKPOINT}" \
    --ckpt_root_path "${OUTPUT_ROOT}/langsplatv2_cluno/legacy_multiscale/mip_nerf360" \
    --output_dir "${OUTPUT_ROOT}/langsplatv2_cluno/eval/mip_nerf360" \
    --mask_thresh 0.4 \
    --json_folder "${DATA_ROOT}/Mip-NeRF360-OVS" \
    --checkpoint "${CHECKPOINT}" \
    --include_feature \
    --topk "${TOPK}" \
    --quick_render
