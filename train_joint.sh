#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(dirname "${REPO_ROOT}")"
WORKSPACE_ROOT="$(dirname "${CODE_ROOT}")"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_ROOT}/Output}"

DATASET_ROOT_PATH="$(realpath "$1")"
DATASET_NAME=$2
INDEX=${3:-0}
DATASET_KIND=${4:-3dovs}
RGB_CHECKPOINT="${RGB_CHECKPOINT:-${OUTPUT_ROOT}/rgb_3dgs/${DATASET_KIND}/${DATASET_NAME}/chkpnt30000.pth}"
PRETRAIN_DIR="${OUTPUT_ROOT}/langsplatv2_cluno/joint/${DATASET_KIND}/${DATASET_NAME}_${INDEX}_pretrain"
MODEL_DIR="${OUTPUT_ROOT}/langsplatv2_cluno/joint/${DATASET_KIND}/${DATASET_NAME}_${INDEX}"
export OUTPUT_ROOT

cd "${REPO_ROOT}"
python train_joint.py \
    -s "${DATASET_ROOT_PATH}/${DATASET_NAME}" \
    -m "${PRETRAIN_DIR}" \
    --start_checkpoint "${RGB_CHECKPOINT}" \
    --feature_levels 1 2 3 \
    --vq_layer_num 1 \
    --codebook_size 64 \
    --iterations 10000 \
    --semantic_only \
    --language_loss_coeff 1.0 \
    --semantic_loss cos \
    --normalize \
    --topk 4

python train_joint.py \
    -s "${DATASET_ROOT_PATH}/${DATASET_NAME}" \
    -m "${MODEL_DIR}" \
    --start_checkpoint "${PRETRAIN_DIR}/chkpnt10000.pth" \
    --reset_iteration \
    --feature_levels 1 2 3 \
    --vq_layer_num 1 \
    --codebook_size 64 \
    --iterations 10000 \
    --semantic_loss cos \
    --normalize \
    --topk 4
