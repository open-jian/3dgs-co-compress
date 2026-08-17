#!/usr/bin/env bash
set -euo pipefail

# Canonical full-joint stage, factored out for machines whose Python location
# differs from the local MVP launcher.  Parameters match run_lerf_scene_mvp.sh.
# Usage: run_lerf_scene_full_joint.sh SCENE [ITERATIONS] [PORT] [SOURCE_CKPT]

SCENE_NAME="${1:?usage: $0 SCENE [ITERATIONS] [PORT] [SOURCE_CKPT]}"
ITERATIONS="${2:-1000}"
PORT="${3:-64600}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/data2/jian/data/lerf_ovs}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/data2/jian/outputs/wacv27_experiments}"
PYTHON_BIN="${PYTHON_BIN:-/home/jian/miniconda3/envs/langsplat_v2/bin/python}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-${EXPERIMENT_ROOT}/clunogs_mvp/lerf_ovs}"
SOURCE_CHECKPOINT="${4:-${PRETRAIN_ROOT}/${SCENE_NAME}/uncompressed/chkpnt${ITERATIONS}.pth}"
RUN_ROOT="${EXPERIMENT_ROOT}/clunogs_mvp/lerf_ovs/${SCENE_NAME}/joint_semantic"

if (( ITERATIONS < 40 )); then
    echo "ITERATIONS must be at least 40 for the pruning/ADMM schedule" >&2
    exit 2
fi
if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "missing three-scale source checkpoint: ${SOURCE_CHECKPOINT}" >&2
    exit 2
fi

mkdir -p "${RUN_ROOT}"
cd "${REPO_ROOT}"

ADMM_START=$(( ITERATIONS / 20 ))
SIMP_FIRST=$(( ITERATIONS / 40 ))
SIMP_SECOND=$(( ITERATIONS * 19 / 20 ))
FREEZE_SH=$(( ITERATIONS * 9 / 10 ))
ADMM_INTERVAL=$(( ITERATIONS * 9 / 20 ))

if [[ ! -f "${RUN_ROOT}/.complete" ]]; then
    /usr/bin/time -v "${PYTHON_BIN}" train_joint.py \
        -s "${DATA_ROOT}/${SCENE_NAME}" \
        -m "${RUN_ROOT}" \
        --start_checkpoint "${SOURCE_CHECKPOINT}" \
        --reset_iteration \
        --feature_levels 1 2 3 \
        --vq_layer_num 1 \
        --codebook_size 64 \
        --iterations "${ITERATIONS}" \
        --semantic_loss cos \
        --normalize \
        --topk 4 \
        --rgb_loss_coeff 1.0 \
        --language_loss_coeff 0.0001 \
        --admm_loss_coeff 1.0 \
        --admm_start_iter "${ADMM_START}" \
        --admm_end_iter "${SIMP_SECOND}" \
        --admm_interval "${ADMM_INTERVAL}" \
        --simp_iteration1 "${SIMP_FIRST}" \
        --simp_iteration2 "${SIMP_SECOND}" \
        --pruning_fraction1 0.00001 \
        --pruning_fraction2 0.5 \
        --rho_opacity 0.0005 \
        --rho_sh 0.0005 \
        --freeze_sh_codebook_iter "${FREEZE_SH}" \
        --sh_codebook_size 256 \
        --port "${PORT}"
    touch "${RUN_ROOT}/.complete"
fi

printf '%s\n' \
    "iterations=${ITERATIONS}" \
    "source_checkpoint=${SOURCE_CHECKPOINT}" \
    "feature_levels=1,2,3" \
    "semantic_codebook_size=64" \
    "semantic_topk=4" \
    "rgb_loss_coeff=1.0" \
    "language_loss_coeff=0.0001" \
    "admm_loss_coeff=1.0" \
    "pruning_fraction1=0.00001" \
    "pruning_fraction2=0.5" \
    "sh_codebook_size=256" \
    "stop_semantic_support_grad=false" \
    > "${RUN_ROOT}/protocol.txt"
