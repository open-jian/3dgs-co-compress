#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(dirname "${REPO_ROOT}")"
WORKSPACE_ROOT="$(dirname "${CODE_ROOT}")"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_ROOT}/Output}"

DATASET_ROOT_PATH="$(realpath "$1")"
DATASET_NAME=$2
INDEX=${3:-0}
CHECKPOINT_ITERATION=${4:-10000}
DATASET_KIND=${5:-3dovs}
LEGACY_ROOT="${OUTPUT_ROOT}/langsplatv2_cluno/legacy_multiscale/${DATASET_KIND}"
JOINT_ROOT="${OUTPUT_ROOT}/langsplatv2_cluno/joint/${DATASET_KIND}"
MERGED_DIR="${JOINT_ROOT}/${DATASET_NAME}_${INDEX}_merged_init"
MODEL_DIR="${JOINT_ROOT}/${DATASET_NAME}_${INDEX}"
export OUTPUT_ROOT

cd "${REPO_ROOT}"
python merge_multiscale_checkpoints.py \
    "${LEGACY_ROOT}/${DATASET_NAME}_${INDEX}_1/chkpnt${CHECKPOINT_ITERATION}.pth" \
    "${LEGACY_ROOT}/${DATASET_NAME}_${INDEX}_2/chkpnt${CHECKPOINT_ITERATION}.pth" \
    "${LEGACY_ROOT}/${DATASET_NAME}_${INDEX}_3/chkpnt${CHECKPOINT_ITERATION}.pth" \
    --output "${MERGED_DIR}/chkpnt${CHECKPOINT_ITERATION}.pth"

python train_joint.py \
    -s "${DATASET_ROOT_PATH}/${DATASET_NAME}" \
    -m "${MODEL_DIR}" \
    --start_checkpoint "${MERGED_DIR}/chkpnt${CHECKPOINT_ITERATION}.pth" \
    --reset_iteration \
    --feature_levels 1 2 3 \
    --vq_layer_num 1 \
    --codebook_size 64 \
    --iterations 10000 \
    --semantic_loss cos \
    --normalize \
    --topk 4
