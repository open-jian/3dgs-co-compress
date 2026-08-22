#!/usr/bin/env bash
set -euo pipefail

SCENE="${1:?scene is required}"
MODEL="${2:?model directory is required}"
GPU="${3:-0}"
ITERATION="${4:-1000}"

REPO=/data2/jian/WACV27_code/ClunoGS
PYTHON=/home/jian/miniconda3/envs/langsplat_v2/bin/python
DATA=/data2/jian/data/lerf_ovs
CHECKPOINT=${MODEL}/chkpnt${ITERATION}.pth
SH_QUANTIZATION=${MODEL}/sh_quantization_${ITERATION}.pth
DEPLOY=${MODEL}/deploy
ARTIFACT=${DEPLOY}/${SCENE}.compact.pth
EVAL=${MODEL}/evaluation

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "missing checkpoint: ${CHECKPOINT}" >&2
    exit 2
fi
if [[ ! -f "${MODEL}/cfg_args" ]]; then
    echo "missing cfg_args: ${MODEL}/cfg_args" >&2
    exit 2
fi

mkdir -p "${DEPLOY}" "${EVAL}"
cp "${MODEL}/cfg_args" "${DEPLOY}/cfg_args"
cd "${REPO}"

EXPORT_ARGS=(
    --checkpoint "${CHECKPOINT}"
    --output "${ARTIFACT}"
    --topk 4
)
if [[ -f "${SH_QUANTIZATION}" ]]; then
    EXPORT_ARGS+=(--sh-quantization "${SH_QUANTIZATION}")
fi
if [[ ! -f "${ARTIFACT}" ]]; then
    "${PYTHON}" compact_artifact.py export "${EXPORT_ARGS[@]}" \
        > "${DEPLOY}/export.json"
fi
"${PYTHON}" compact_artifact.py validate --artifact "${ARTIFACT}" \
    > "${DEPLOY}/validation.json"

if [[ ! -f "${EVAL}/semantic/${SCENE}_0/metrics_lerf.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" eval_lerf.py \
        -s "${DATA}/${SCENE}" \
        -m "${DEPLOY}" \
        --dataset_name "${SCENE}" \
        --index 0 \
        --compact_artifact "${ARTIFACT}" \
        --ckpt_root_path "${DEPLOY}" \
        --output_dir "${EVAL}/semantic" \
        --mask_thresh 0.4 \
        --json_folder "${DATA}/label" \
        --checkpoint "${ITERATION}" \
        --include_feature \
        --topk 4 \
        --quick_render
fi

if [[ ! -f "${EVAL}/rgb.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" eval_compact_rgb.py \
        -s "${DATA}/${SCENE}" \
        -m "${DEPLOY}" \
        --eval \
        --compact_artifact "${ARTIFACT}" \
        --output "${EVAL}/rgb.json"
fi

printf '%s\n' \
    "fresh_process_reload=true" \
    "checkpoint=${CHECKPOINT}" \
    "sh_quantization=$([[ -f ${SH_QUANTIZATION} ]] && echo ${SH_QUANTIZATION} || echo none)" \
    "compact_artifact=${ARTIFACT}" \
    "artifact_bytes=$(stat -c %s "${ARTIFACT}")" \
    > "${MODEL}/fresh_reload_provenance.txt"
touch "${MODEL}/.finalized"
