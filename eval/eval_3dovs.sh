#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_paths.sh"

CASE_NAME=$1
python "${EVAL_DIR}/eval_3dovs.py" \
    --dataset_name "${CASE_NAME}" \
    --feat_dir "${COLASPLAT_OUTPUT}/semantic/3dovs" \
    --ae_ckpt_dir "${AE_CKPT_ROOT}" \
    --output_dir "${COLASPLAT_OUTPUT}/eval/semantic/3dovs" \
    --mask_thresh 0.4 \
    --encoder_dims 256 128 64 32 3 \
    --decoder_dims 16 32 64 128 256 256 512 \
    --dataset_path "${DATA_ROOT}/3dovs"
