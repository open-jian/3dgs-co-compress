#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_paths.sh"

CASE_NAME=$1
python "${EVAL_DIR}/eval_lerf.py" \
    --dataset_name "${CASE_NAME}" \
    --feat_dir "${COLASPLAT_OUTPUT}/semantic/lerf_ovs" \
    --ae_ckpt_dir "${AE_CKPT_ROOT}" \
    --output_dir "${COLASPLAT_OUTPUT}/eval/semantic/lerf_ovs" \
    --mask_thresh 0.4 \
    --encoder_dims 256 128 64 32 3 \
    --decoder_dims 16 32 64 128 256 256 512 \
    --json_folder "${DATA_ROOT}/lerf_ovs/label"
