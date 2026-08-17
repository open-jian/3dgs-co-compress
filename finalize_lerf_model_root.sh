#!/usr/bin/env bash
set -euo pipefail

# Export, validate, and evaluate one arbitrary finished model directory.
# Every operation is a separate Python process; evaluation therefore reloads
# only the counted compact artifact and cannot retain the training checkpoint.
#
# Usage: finalize_lerf_model_root.sh SCENE MODEL_ROOT [GPU] [ITERATION]

SCENE_NAME="${1:?SCENE is required}"
MODEL_ROOT="${2:?MODEL_ROOT is required}"
GPU_ID="${3:-0}"
ITERATION="${4:-1000}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/jian/miniconda3/envs/langsplat_v2/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data2/jian/data/lerf_ovs}"
CHECKPOINT="${MODEL_ROOT}/chkpnt${ITERATION}.pth"
SH_QUANTIZATION="${MODEL_ROOT}/sh_quantization_${ITERATION}.pth"
ARTIFACT_ROOT="${MODEL_ROOT}/deploy"
ARTIFACT="${ARTIFACT_ROOT}/${SCENE_NAME}.compact.pth"
EVAL_ROOT="${MODEL_ROOT}/evaluation"

if [[ ! -f "${CHECKPOINT}" || ! -e "${SH_QUANTIZATION}" ]]; then
    echo "missing checkpoint or SH quantization in ${MODEL_ROOT}" >&2
    exit 2
fi

mkdir -p "${ARTIFACT_ROOT}" "${EVAL_ROOT}"
if [[ -f "${MODEL_ROOT}/cfg_args" ]]; then
    cp "${MODEL_ROOT}/cfg_args" "${ARTIFACT_ROOT}/cfg_args"
else
    echo "missing cfg_args in ${MODEL_ROOT}" >&2
    exit 2
fi

cd "${REPO_ROOT}"
if [[ ! -f "${ARTIFACT}" ]]; then
    "${PYTHON_BIN}" compact_artifact.py export \
        --checkpoint "${CHECKPOINT}" \
        --sh-quantization "${SH_QUANTIZATION}" \
        --output "${ARTIFACT}" \
        --topk 4 \
        > "${ARTIFACT_ROOT}/export.json"
fi

# Separate process one: structural/hash validation from disk.
"${PYTHON_BIN}" compact_artifact.py validate --artifact "${ARTIFACT}" \
    > "${ARTIFACT_ROOT}/validation.json"

# Separate process two: fresh compact-only semantic reload and LERF evaluation.
if [[ ! -f "${EVAL_ROOT}/semantic/${SCENE_NAME}_0/metrics_lerf.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_lerf.py \
        -s "${DATA_ROOT}/${SCENE_NAME}" \
        -m "${ARTIFACT_ROOT}" \
        --dataset_name "${SCENE_NAME}" \
        --index 0 \
        --compact_artifact "${ARTIFACT}" \
        --ckpt_root_path "${ARTIFACT_ROOT}" \
        --output_dir "${EVAL_ROOT}/semantic" \
        --mask_thresh 0.4 \
        --json_folder "${DATA_ROOT}/label" \
        --checkpoint "${ITERATION}" \
        --include_feature \
        --topk 4 \
        --quick_render \
        --save_visuals
fi

# Separate process three: fresh compact-only held-out RGB evaluation.
if [[ ! -f "${EVAL_ROOT}/rgb.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_compact_rgb.py \
        -s "${DATA_ROOT}/${SCENE_NAME}" \
        -m "${ARTIFACT_ROOT}" \
        --eval \
        --compact_artifact "${ARTIFACT}" \
        --output "${EVAL_ROOT}/rgb.json"
fi

printf '%s\n' \
    "fresh_process_reload=true" \
    "checkpoint=${CHECKPOINT}" \
    "sh_quantization=${SH_QUANTIZATION}" \
    "compact_artifact=${ARTIFACT}" \
    "artifact_bytes=$(stat -c %s "${ARTIFACT}")" \
    "semantic_metrics=${EVAL_ROOT}/semantic/${SCENE_NAME}_0/metrics_lerf.json" \
    "rgb_metrics=${EVAL_ROOT}/rgb.json" \
    > "${MODEL_ROOT}/fresh_reload_provenance.txt"
touch "${MODEL_ROOT}/.finalized"
