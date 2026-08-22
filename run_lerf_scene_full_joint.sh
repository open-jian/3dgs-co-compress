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
RUN_ROOT="${RUN_ROOT:-${EXPERIMENT_ROOT}/clunogs_mvp/lerf_ovs/${SCENE_NAME}/joint_semantic}"

# Explicit C3DGS-ADMM settings. Environment overrides make compression curves
# reproducible without editing this launcher.
RHO_COLOR="${RHO_COLOR:-0.0005}"
RHO_COVARIANCE="${RHO_COVARIANCE:-0.0005}"
RHO_SEMANTIC="${RHO_SEMANTIC:-0.0005}"
COLOR_CODEBOOK_SIZE="${COLOR_CODEBOOK_SIZE:-256}"
COVARIANCE_CODEBOOK_SIZE="${COVARIANCE_CODEBOOK_SIZE:-256}"
SEMANTIC_CODEBOOK_SIZE="${SEMANTIC_CODEBOOK_SIZE:-256}"
C3DGS_CODEBOOK_DECAY="${C3DGS_CODEBOOK_DECAY:-0.8}"
C3DGS_SENSITIVITY_DECAY="${C3DGS_SENSITIVITY_DECAY:-0.9}"
C3DGS_KEEP_RATIO="${C3DGS_KEEP_RATIO:-0.01}"
C3DGS_REFINEMENT_STEPS="${C3DGS_REFINEMENT_STEPS:-100}"
C3DGS_COVARIANCE_REFINEMENT_STEPS="${C3DGS_COVARIANCE_REFINEMENT_STEPS:-800}"
C3DGS_SEMANTIC_REFINEMENT_STEPS="${C3DGS_SEMANTIC_REFINEMENT_STEPS:-100}"
C3DGS_CHUNK_SIZE="${C3DGS_CHUNK_SIZE:-4096}"

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
        --rho_sh "${RHO_COLOR}" \
        --rho_covariance "${RHO_COVARIANCE}" \
        --rho_semantic "${RHO_SEMANTIC}" \
        --freeze_sh_codebook_iter "${FREEZE_SH}" \
        --sh_codebook_size "${COLOR_CODEBOOK_SIZE}" \
        --gaussian_codebook_size "${COVARIANCE_CODEBOOK_SIZE}" \
        --semantic_coefficient_codebook_size "${SEMANTIC_CODEBOOK_SIZE}" \
        --c3dgs_codebook_decay "${C3DGS_CODEBOOK_DECAY}" \
        --c3dgs_sensitivity_decay "${C3DGS_SENSITIVITY_DECAY}" \
        --c3dgs_keep_ratio "${C3DGS_KEEP_RATIO}" \
        --c3dgs_refinement_steps "${C3DGS_REFINEMENT_STEPS}" \
        --c3dgs_covariance_refinement_steps "${C3DGS_COVARIANCE_REFINEMENT_STEPS}" \
        --c3dgs_semantic_refinement_steps "${C3DGS_SEMANTIC_REFINEMENT_STEPS}" \
        --c3dgs_chunk_size "${C3DGS_CHUNK_SIZE}" \
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
    "quantization=c3dgs_sensitivity_aware_vq_admm" \
    "color_codebook_size=${COLOR_CODEBOOK_SIZE}" \
    "covariance_codebook_size=${COVARIANCE_CODEBOOK_SIZE}" \
    "semantic_coefficient_codebook_size=${SEMANTIC_CODEBOOK_SIZE}" \
    "rho_color=${RHO_COLOR}" \
    "rho_covariance=${RHO_COVARIANCE}" \
    "rho_semantic=${RHO_SEMANTIC}" \
    "c3dgs_codebook_decay=${C3DGS_CODEBOOK_DECAY}" \
    "c3dgs_sensitivity_decay=${C3DGS_SENSITIVITY_DECAY}" \
    "c3dgs_keep_ratio=${C3DGS_KEEP_RATIO}" \
    "c3dgs_refinement_steps=${C3DGS_REFINEMENT_STEPS}" \
    "c3dgs_covariance_refinement_steps=${C3DGS_COVARIANCE_REFINEMENT_STEPS}" \
    "c3dgs_semantic_refinement_steps=${C3DGS_SEMANTIC_REFINEMENT_STEPS}" \
    "c3dgs_chunk_size=${C3DGS_CHUNK_SIZE}" \
    "stop_semantic_support_grad=false" \
    > "${RUN_ROOT}/protocol.txt"
