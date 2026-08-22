#!/usr/bin/env bash
set -euo pipefail

SCENE="${1:?scene is required}"
GPU="${2:?gpu is required}"
TOTAL_ITERATIONS="${3:-1000}"
PORT_BASE="${4:-55400}"

REPO=/data2/jian/WACV27_code/ClunoGS
PYTHON=/home/jian/miniconda3/envs/langsplat_v2/bin/python
DATA=/data2/jian/data/lerf_ovs
SOURCE=/data2/jian/outputs/wacv27_experiments/clunogs_mvp/lerf_ovs/${SCENE}/uncompressed/chkpnt1000.pth
OUTPUT_ROOT="${CONTROL_OUTPUT_ROOT:-/data2/jian/outputs/wacv27_section6_20260819}"
RUN=${OUTPUT_ROOT}/vq_then_spars/${SCENE}
VQ_DIR=${RUN}/vq_first
SPARS_DIR=${RUN}/spars_second
VQ_ITERATIONS=$(( TOTAL_ITERATIONS / 2 ))
SPARS_ITERATIONS=$(( TOTAL_ITERATIONS - VQ_ITERATIONS ))

if [[ ! -f "${SOURCE}" ]]; then
    echo "missing source checkpoint: ${SOURCE}" >&2
    exit 2
fi
if (( VQ_ITERATIONS < 40 || SPARS_ITERATIONS < 40 )); then
    echo "each sequential stage must have at least 40 iterations" >&2
    exit 2
fi
mkdir -p "${VQ_DIR}" "${SPARS_DIR}"
cd "${REPO}"

vq_admm_start=$(( VQ_ITERATIONS / 20 ))
vq_admm_end=$(( VQ_ITERATIONS * 19 / 20 ))
vq_interval=$(( VQ_ITERATIONS * 9 / 20 ))
vq_freeze=$(( VQ_ITERATIONS * 9 / 10 ))

if [[ ! -f "${VQ_DIR}/.complete" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" /usr/bin/time -v "${PYTHON}" train_joint.py \
        -s "${DATA}/${SCENE}" \
        -m "${VQ_DIR}" \
        --start_checkpoint "${SOURCE}" \
        --reset_iteration \
        --feature_levels 1 2 3 \
        --vq_layer_num 1 \
        --codebook_size 64 \
        --iterations "${VQ_ITERATIONS}" \
        --semantic_loss cos \
        --normalize \
        --topk 4 \
        --rgb_loss_coeff 1.0 \
        --language_loss_coeff 0.0001 \
        --admm_loss_coeff 1.0 \
        --admm_start_iter "${vq_admm_start}" \
        --admm_end_iter "${vq_admm_end}" \
        --admm_interval "${vq_interval}" \
        --simp_iteration1 1 \
        --simp_iteration2 "${vq_admm_end}" \
        --pruning_fraction1 0 \
        --pruning_fraction2 0 \
        --rho_opacity 0.0005 \
        --rho_sh 0.0005 \
        --freeze_sh_codebook_iter "${vq_freeze}" \
        --sh_codebook_size 256 \
        --track_source_ids \
        --materialize_sh_projection_at_checkpoint \
        --port "${PORT_BASE}"
    touch "${VQ_DIR}/.complete"
fi

VQ_CHECKPOINT=${VQ_DIR}/chkpnt${VQ_ITERATIONS}.pth
VQ_LINEAGE=${VQ_CHECKPOINT}.source_ids.pt
VQ_QUANT=${VQ_DIR}/sh_quantization_${VQ_ITERATIONS}.pth
for required in "${VQ_CHECKPOINT}" "${VQ_LINEAGE}" "${VQ_QUANT}"; do
    [[ -f "${required}" ]] || { echo "missing VQ-first output: ${required}" >&2; exit 3; }
done

spars_admm_start=$(( SPARS_ITERATIONS / 20 ))
spars_admm_end=$(( SPARS_ITERATIONS * 19 / 20 ))
spars_interval=$(( SPARS_ITERATIONS * 9 / 20 ))

if [[ ! -f "${SPARS_DIR}/.complete" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" /usr/bin/time -v "${PYTHON}" train_joint.py \
        -s "${DATA}/${SCENE}" \
        -m "${SPARS_DIR}" \
        --start_checkpoint "${VQ_CHECKPOINT}" \
        --reset_iteration \
        --feature_levels 1 2 3 \
        --vq_layer_num 1 \
        --codebook_size 64 \
        --iterations "${SPARS_ITERATIONS}" \
        --support_only \
        --semantic_loss cos \
        --normalize \
        --topk 4 \
        --rgb_loss_coeff 1.0 \
        --language_loss_coeff 0.0001 \
        --admm_loss_coeff 1.0 \
        --admm_start_iter "${spars_admm_start}" \
        --admm_end_iter "${spars_admm_end}" \
        --admm_interval "${spars_interval}" \
        --simp_iteration1 $(( SPARS_ITERATIONS / 40 )) \
        --simp_iteration2 "${spars_admm_end}" \
        --pruning_fraction1 0.00001 \
        --pruning_fraction2 0.5 \
        --rho_opacity 0.0005 \
        --rho_sh 0.0005 \
        --freeze_sh_codebook_iter "${spars_admm_end}" \
        --sh_codebook_size 256 \
        --disable_sh_admm \
        --freeze_higher_order_sh \
        --track_source_ids \
        --source_id_input "${VQ_LINEAGE}" \
        --port "$(( PORT_BASE + 20 ))"
    touch "${SPARS_DIR}/.complete"
fi

FINAL_CHECKPOINT=${SPARS_DIR}/chkpnt${SPARS_ITERATIONS}.pth
FINAL_LINEAGE=${FINAL_CHECKPOINT}.source_ids.pt
FINAL_QUANT=${SPARS_DIR}/sh_quantization_${SPARS_ITERATIONS}.pth
if [[ ! -f "${FINAL_QUANT}" ]]; then
    "${PYTHON}" subset_sh_quantization.py \
        --source-quantization "${VQ_QUANT}" \
        --lineage-sidecar "${FINAL_LINEAGE}" \
        --output "${FINAL_QUANT}"
fi

printf '%s\n' \
    "order=vq_then_sparsification" \
    "total_iterations=${TOTAL_ITERATIONS}" \
    "vq_iterations=${VQ_ITERATIONS}" \
    "sparsification_iterations=${SPARS_ITERATIONS}" \
    "source_checkpoint=${SOURCE}" \
    "materialized_vq=true" \
    "higher_order_sh_frozen_in_stage_two=true" \
    > "${RUN}/protocol.txt"
touch "${RUN}/.pipeline.complete"
