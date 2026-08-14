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
MODEL_DIR="${OUTPUT_ROOT}/langsplatv2_cluno/joint/3dovs/${DATASET_NAME}_${INDEX}"
JOINT_CHECKPOINT="${JOINT_CHECKPOINT:-${MODEL_DIR}/chkpnt${CHECKPOINT}.pth}"

cd "${REPO_ROOT}"
python eval_3d_ovs.py \
    -s "${DATA_ROOT}/3dovs/${DATASET_NAME}" \
    -m "${MODEL_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --index "${INDEX}" \
    --joint_checkpoint "${JOINT_CHECKPOINT}" \
    --ckpt_root_path "${OUTPUT_ROOT}/langsplatv2_cluno/legacy_multiscale/3dovs" \
    --output_dir "${OUTPUT_ROOT}/langsplatv2_cluno/eval/3dovs" \
    --mask_thresh 0.25 \
    --gt_mask_dir "${DATA_ROOT}/3dovs/${DATASET_NAME}/segmentations" \
    --checkpoint "${CHECKPOINT}" \
    --include_feature \
    --topk 4 \
    --quick_render
