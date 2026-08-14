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
TOPK=4
RGB_CHECKPOINT="${RGB_CHECKPOINT:-${OUTPUT_ROOT}/rgb_3dgs/${DATASET_KIND}/${DATASET_NAME}/chkpnt30000.pth}"
LEGACY_ROOT="${OUTPUT_ROOT}/langsplatv2_cluno/legacy_multiscale/${DATASET_KIND}"
export OUTPUT_ROOT

cd "${REPO_ROOT}"
for level in 1 2 3; do
    python train.py \
        -s "${DATASET_ROOT_PATH}/${DATASET_NAME}" \
        -m "${LEGACY_ROOT}/${DATASET_NAME}_${INDEX}" \
        --start_checkpoint "${RGB_CHECKPOINT}" \
        --feature_level "${level}" \
        --vq_layer_num 1 \
        --codebook_size 64 \
        --cos_loss \
        --topk "${TOPK}"
done
