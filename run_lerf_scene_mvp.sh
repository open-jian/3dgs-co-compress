#!/usr/bin/env bash
set -euo pipefail

SCENE_NAME="$1"
ITERATIONS="${2:-1000}"
PORT_BASE="${3:-61000}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="/data2/jian/data/lerf_ovs"
RGB_ROOT="/data2/jian/outputs/rgb_3dgs_imported_checkpoints/lerf_ovs"
RUN_ROOT="/data2/jian/outputs/wacv27_experiments/clunogs_mvp/lerf_ovs/${SCENE_NAME}"
PYTHON_BIN="/home/jian/miniconda3/envs/langsplat_v2/bin/python"
PRETRAIN_DIR="${RUN_ROOT}/uncompressed"
JOINT_DIR="${RUN_ROOT}/joint_semantic"
RGB_CHECKPOINT="${RGB_ROOT}/${SCENE_NAME}/chkpnt30000.pth"

mkdir -p "${PRETRAIN_DIR}" "${JOINT_DIR}"
cd "${REPO_ROOT}"

if [[ ! -f "${PRETRAIN_DIR}/.complete" ]]; then
    /usr/bin/time -v "${PYTHON_BIN}" train_joint.py \
        -s "${DATA_ROOT}/${SCENE_NAME}" \
        -m "${PRETRAIN_DIR}" \
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
        --port "${PORT_BASE}"
    touch "${PRETRAIN_DIR}/.complete"
fi

if [[ ! -f "${JOINT_DIR}/.complete" ]]; then
    ADMM_START=$(( ITERATIONS / 20 ))
    SIMP_FIRST=$(( ITERATIONS / 40 ))
    SIMP_SECOND=$(( ITERATIONS * 19 / 20 ))
    FREEZE_SH=$(( ITERATIONS * 9 / 10 ))
    ADMM_INTERVAL=$(( ITERATIONS * 9 / 20 ))
    /usr/bin/time -v "${PYTHON_BIN}" train_joint.py \
        -s "${DATA_ROOT}/${SCENE_NAME}" \
        -m "${JOINT_DIR}" \
        --start_checkpoint "${PRETRAIN_DIR}/chkpnt${ITERATIONS}.pth" \
        --reset_iteration \
        --feature_levels 1 2 3 \
        --vq_layer_num 1 \
        --codebook_size 64 \
        --iterations "${ITERATIONS}" \
        --semantic_loss cos \
        --normalize \
        --topk 4 \
        --admm_start_iter "${ADMM_START}" \
        --admm_end_iter "${SIMP_SECOND}" \
        --admm_interval "${ADMM_INTERVAL}" \
        --simp_iteration1 "${SIMP_FIRST}" \
        --simp_iteration2 "${SIMP_SECOND}" \
        --freeze_sh_codebook_iter "${FREEZE_SH}" \
        --sh_codebook_size 256 \
        --port "$(( PORT_BASE + 100 ))"
    touch "${JOINT_DIR}/.complete"
fi

touch "${RUN_ROOT}/.pipeline.complete"
