#!/usr/bin/env bash
set -euo pipefail

SCENE_NAME="$1"
GPU_ID="${2:-0}"
VARIANT="${3:-joint_semantic}"
ITERATION="${4:-1000}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/jian/miniconda3/envs/langsplat_v2/bin/python"
DATA_ROOT="/data2/jian/data/lerf_ovs"
RUN_ROOT="/data2/jian/outputs/wacv27_experiments/clunogs_mvp/lerf_ovs/${SCENE_NAME}"
MODEL_ROOT="${RUN_ROOT}/${VARIANT}"
CHECKPOINT="${MODEL_ROOT}/chkpnt${ITERATION}.pth"
SH_QUANTIZATION="${MODEL_ROOT}/sh_quantization_${ITERATION}.pth"
ARTIFACT_ROOT="${MODEL_ROOT}/deploy"
ARTIFACT="${ARTIFACT_ROOT}/${SCENE_NAME}.compact.pth"
EVAL_ROOT="${MODEL_ROOT}/evaluation"

if [[ ! -f "${CHECKPOINT}" || ! -f "${SH_QUANTIZATION}" ]]; then
    printf 'missing final checkpoint or SH quantization for %s/%s\n' "${SCENE_NAME}" "${VARIANT}" >&2
    exit 2
fi

mkdir -p "${ARTIFACT_ROOT}" "${EVAL_ROOT}"
cp "${MODEL_ROOT}/cfg_args" "${ARTIFACT_ROOT}/cfg_args"
cd "${REPO_ROOT}"
if [[ ! -f "${ARTIFACT}" ]]; then
    "${PYTHON_BIN}" compact_artifact.py export \
        --checkpoint "${CHECKPOINT}" \
        --sh-quantization "${SH_QUANTIZATION}" \
        --output "${ARTIFACT}" \
        --topk 4
fi
"${PYTHON_BIN}" compact_artifact.py validate --artifact "${ARTIFACT}" \
    > "${ARTIFACT_ROOT}/validation.json"

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
        --quick_render
fi

if [[ ! -f "${EVAL_ROOT}/rgb.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_compact_rgb.py \
        -s "${DATA_ROOT}/${SCENE_NAME}" \
        -m "${ARTIFACT_ROOT}" \
        --eval \
        --compact_artifact "${ARTIFACT}" \
        --output "${EVAL_ROOT}/rgb.json"
fi

touch "${MODEL_ROOT}/.finalized"
