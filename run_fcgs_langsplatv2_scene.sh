#!/usr/bin/env bash
set -euo pipefail

SCENE_NAME="$1"
ITERATIONS="${2:-1000}"
PORT="${3:-63200}"
OPERATING_POINT="${4:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x /home/jian/miniconda3/envs/langsplat_v2/bin/python ]]; then
        PYTHON_BIN=/home/jian/miniconda3/envs/langsplat_v2/bin/python
    elif [[ -x /home/jian/.local/share/mamba/envs/langsplat_v2/bin/python ]]; then
        PYTHON_BIN=/home/jian/.local/share/mamba/envs/langsplat_v2/bin/python
    else
        echo "cannot find the langsplat_v2 Python interpreter" >&2
        exit 2
    fi
fi
DATA_ROOT="/data2/jian/data/lerf_ovs"
RGB_CHECKPOINT="/data2/jian/outputs/wacv27_experiments/fcgs_imported_checkpoints/lerf_ovs/${SCENE_NAME}/chkpnt30000.pth"
OUTPUT_ROOT="/data2/jian/outputs/wacv27_experiments/fcgs_langsplatv2/lerf_ovs/${SCENE_NAME}"
if [[ -n "${OPERATING_POINT}" ]]; then
    RGB_CHECKPOINT="/data2/jian/outputs/wacv27_experiments/fcgs_imported_checkpoints/lerf_ovs/${SCENE_NAME}/${OPERATING_POINT}/chkpnt30000.pth"
    OUTPUT_ROOT="${OUTPUT_ROOT}/${OPERATING_POINT}"
fi

if [[ ! -f "${RGB_CHECKPOINT}" ]]; then
    echo "missing decoded FCGS checkpoint: ${RGB_CHECKPOINT}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

if [[ ! -f "${OUTPUT_ROOT}/.complete" ]]; then
    /usr/bin/time -v "${PYTHON_BIN}" train_joint.py \
        -s "${DATA_ROOT}/${SCENE_NAME}" \
        -m "${OUTPUT_ROOT}" \
        --start_checkpoint "${RGB_CHECKPOINT}" \
        --feature_levels 1 2 3 \
        --vq_layer_num 1 \
        --codebook_size 64 \
        --iterations "${ITERATIONS}" \
        --semantic_only \
        --language_loss_coeff 1.0 \
        --semantic_loss cos \
        --normalize \
        --topk 4 \
        --port "${PORT}"
    touch "${OUTPUT_ROOT}/.complete"
fi
