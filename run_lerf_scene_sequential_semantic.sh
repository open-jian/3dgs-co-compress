#!/usr/bin/env bash
set -euo pipefail

# Matched sequential semantic-support ablation:
#   1) sparsify support with RGB + semantic reconstruction while keeping the
#      semantic logits/codebooks fixed, then
#   2) freeze that retained support and optimize the semantic VQ parameters.
# The two stages split the same total optimizer-update budget as joint.

SCENE_NAME="${1:?usage: $0 SCENE [TOTAL_ITERS] [PORT_BASE] [SOURCE_CKPT]}"
TOTAL_ITERATIONS="${2:-1000}"
PORT_BASE="${3:-67000}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/data2/jian/data/lerf_ovs}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/data2/jian/outputs/wacv27_experiments}"
PYTHON_BIN="${PYTHON_BIN:-/home/jian/miniconda3/envs/langsplat_v2/bin/python}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-${EXPERIMENT_ROOT}/clunogs_mvp/lerf_ovs}"
SOURCE_CHECKPOINT="${4:-${PRETRAIN_ROOT}/${SCENE_NAME}/uncompressed/chkpnt${TOTAL_ITERATIONS}.pth}"
LANGUAGE_LOSS_COEFF="${LANGUAGE_LOSS_COEFF:-0.0001}"

SUPPORT_ITERATIONS="${SEQUENTIAL_SUPPORT_ITERATIONS:-$(( TOTAL_ITERATIONS / 2 ))}"
SEMANTIC_ITERATIONS=$(( TOTAL_ITERATIONS - SUPPORT_ITERATIONS ))
RUN_ROOT="${EXPERIMENT_ROOT}/clunogs_sequential_semantic/lerf_ovs/${SCENE_NAME}"
SUPPORT_DIR="${RUN_ROOT}/semantic_guided_support"
SEMANTIC_DIR="${RUN_ROOT}/semantic_vq"

if (( TOTAL_ITERATIONS < 40 )); then
    echo "TOTAL_ITERS must be at least 40 for the pruning/ADMM schedule" >&2
    exit 2
fi
if (( SUPPORT_ITERATIONS < 20 || SEMANTIC_ITERATIONS < 1 )); then
    echo "invalid split: support=${SUPPORT_ITERATIONS}, semantic=${SEMANTIC_ITERATIONS}" >&2
    exit 2
fi
if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "missing three-scale source checkpoint: ${SOURCE_CHECKPOINT}" >&2
    exit 2
fi

mkdir -p "${SUPPORT_DIR}" "${SEMANTIC_DIR}"
cd "${REPO_ROOT}"

ADMM_START=$(( SUPPORT_ITERATIONS / 20 ))
SIMP_FIRST=$(( SUPPORT_ITERATIONS / 40 ))
SIMP_SECOND=$(( SUPPORT_ITERATIONS * 19 / 20 ))
FREEZE_SH=$(( SUPPORT_ITERATIONS * 9 / 10 ))
ADMM_INTERVAL=$(( SUPPORT_ITERATIONS * 9 / 20 ))

if [[ ! -f "${SUPPORT_DIR}/.complete" ]]; then
    /usr/bin/time -v "${PYTHON_BIN}" train_joint.py \
        -s "${DATA_ROOT}/${SCENE_NAME}" \
        -m "${SUPPORT_DIR}" \
        --start_checkpoint "${SOURCE_CHECKPOINT}" \
        --reset_iteration \
        --feature_levels 1 2 3 \
        --vq_layer_num 1 \
        --codebook_size 64 \
        --iterations "${SUPPORT_ITERATIONS}" \
        --support_only \
        --language_loss_coeff "${LANGUAGE_LOSS_COEFF}" \
        --semantic_loss cos \
        --normalize \
        --topk 4 \
        --rgb_loss_coeff 1.0 \
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
        --port "${PORT_BASE}"
    touch "${SUPPORT_DIR}/.complete"
fi

SUPPORT_CHECKPOINT="${SUPPORT_DIR}/chkpnt${SUPPORT_ITERATIONS}.pth"
if [[ ! -f "${SUPPORT_CHECKPOINT}" ]]; then
    echo "missing semantic-guided support checkpoint: ${SUPPORT_CHECKPOINT}" >&2
    exit 3
fi

if [[ ! -f "${SEMANTIC_DIR}/.complete" ]]; then
    /usr/bin/time -v "${PYTHON_BIN}" train_joint.py \
        -s "${DATA_ROOT}/${SCENE_NAME}" \
        -m "${SEMANTIC_DIR}" \
        --start_checkpoint "${SUPPORT_CHECKPOINT}" \
        --reset_iteration \
        --feature_levels 1 2 3 \
        --vq_layer_num 1 \
        --codebook_size 64 \
        --iterations "${SEMANTIC_ITERATIONS}" \
        --semantic_only \
        --language_loss_coeff "${LANGUAGE_LOSS_COEFF}" \
        --semantic_loss cos \
        --normalize \
        --topk 4 \
        --port "$(( PORT_BASE + 100 ))"
    touch "${SEMANTIC_DIR}/.complete"
fi

SUPPORT_SH="${SUPPORT_DIR}/sh_quantization_${SUPPORT_ITERATIONS}.pth"
FINAL_SH="${SEMANTIC_DIR}/sh_quantization_${SEMANTIC_ITERATIONS}.pth"
if [[ -f "${SUPPORT_SH}" && ! -e "${FINAL_SH}" ]]; then
    ln -s "${SUPPORT_SH}" "${FINAL_SH}"
fi

printf '%s\n' \
    "support_guidance=semantic" \
    "support_semantics=frozen" \
    "total_iterations=${TOTAL_ITERATIONS}" \
    "support_iterations=${SUPPORT_ITERATIONS}" \
    "semantic_iterations=${SEMANTIC_ITERATIONS}" \
    "language_loss_coeff=${LANGUAGE_LOSS_COEFF}" \
    "source_checkpoint=${SOURCE_CHECKPOINT}" \
    > "${RUN_ROOT}/budget.txt"
touch "${RUN_ROOT}/.pipeline.complete"
