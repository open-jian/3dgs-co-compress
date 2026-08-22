#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?variant: spars_only | vq_only | no_dual | full_tracked | no_sem_tracked}"
SCENE="${2:?scene is required}"
GPU="${3:?gpu is required}"
ITERATIONS="${4:-1000}"
PORT="${5:-55000}"

REPO=/data2/jian/WACV27_code/ClunoGS
PYTHON=/home/jian/miniconda3/envs/langsplat_v2/bin/python
DATA=/data2/jian/data/lerf_ovs
SOURCE=/data2/jian/outputs/wacv27_experiments/clunogs_mvp/lerf_ovs/${SCENE}/uncompressed/chkpnt1000.pth
OUTPUT_ROOT="${CONTROL_OUTPUT_ROOT:-/data2/jian/outputs/wacv27_section6_20260819}"
MODEL=${OUTPUT_ROOT}/${VARIANT}/${SCENE}

if [[ ! -f "${SOURCE}" ]]; then
    echo "missing source checkpoint: ${SOURCE}" >&2
    exit 2
fi
if (( ITERATIONS < 40 )); then
    echo "iterations must be at least 40" >&2
    exit 2
fi

mkdir -p "${MODEL}"
cd "${REPO}"

ADMM_START=$(( ITERATIONS / 20 ))
SIMP_FIRST=$(( ITERATIONS / 40 ))
SIMP_SECOND=$(( ITERATIONS * 19 / 20 ))
FREEZE_SH=$(( ITERATIONS * 9 / 10 ))
ADMM_INTERVAL=$(( ITERATIONS * 9 / 20 ))

EXTRA=()
PRUNE_FIRST=0.00001
PRUNE_SECOND=0.5
case "${VARIANT}" in
    spars_only)
        EXTRA+=(--disable_sh_admm)
        ;;
    vq_only)
        PRUNE_FIRST=0
        PRUNE_SECOND=0
        ;;
    no_dual)
        EXTRA+=(--disable_dual_updates)
        ;;
    full_tracked)
        EXTRA+=(--track_source_ids)
        ;;
    no_sem_tracked)
        EXTRA+=(--stop_semantic_support_grad --track_source_ids)
        ;;
    *)
        echo "unknown variant: ${VARIANT}" >&2
        exit 2
        ;;
esac

if [[ ! -f "${MODEL}/.complete" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" /usr/bin/time -v "${PYTHON}" train_joint.py \
        -s "${DATA}/${SCENE}" \
        -m "${MODEL}" \
        --start_checkpoint "${SOURCE}" \
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
        --pruning_fraction1 "${PRUNE_FIRST}" \
        --pruning_fraction2 "${PRUNE_SECOND}" \
        --rho_opacity 0.0005 \
        --rho_sh 0.0005 \
        --freeze_sh_codebook_iter "${FREEZE_SH}" \
        --sh_codebook_size 256 \
        --port "${PORT}" \
        "${EXTRA[@]}"
    touch "${MODEL}/.complete"
fi

printf '%s\n' \
    "variant=${VARIANT}" \
    "scene=${SCENE}" \
    "iterations=${ITERATIONS}" \
    "source_checkpoint=${SOURCE}" \
    "pruning_fraction1=${PRUNE_FIRST}" \
    "pruning_fraction2=${PRUNE_SECOND}" \
    "disable_sh_admm=$([[ ${VARIANT} == spars_only ]] && echo true || echo false)" \
    "disable_dual_updates=$([[ ${VARIANT} == no_dual ]] && echo true || echo false)" \
    "stop_semantic_support_grad=$([[ ${VARIANT} == no_sem_tracked ]] && echo true || echo false)" \
    "track_source_ids=$([[ ${VARIANT} == full_tracked || ${VARIANT} == no_sem_tracked ]] && echo true || echo false)" \
    > "${MODEL}/protocol.txt"
