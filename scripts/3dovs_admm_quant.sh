#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="$(dirname "${REPO_ROOT}")"
WORKSPACE_ROOT="$(dirname "${CODE_ROOT}")"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_ROOT}/Output}"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE_ROOT}/Data}"
SEMANTIC_ROOT="${OUTPUT_ROOT}/colasplat/semantic/3dovs"
MODEL_ROOT="${OUTPUT_ROOT}/colasplat/admm_quant/3dovs"
LOG_ROOT="${OUTPUT_ROOT}/colasplat/logs/nohup_logs"
export OUTPUT_ROOT DATA_ROOT

casenames=("bed")
gpu_pool=(0 1 2)
mkdir -p "${LOG_ROOT}/train_admm_quant" "${LOG_ROOT}/render_admm_quant" \
    "${LOG_ROOT}/eval_admm_quant"
cd "${REPO_ROOT}"

for i in "${!casenames[@]}"; do
    {
        casename=${casenames[$i]}
        gpu_id=${gpu_pool[$i]}
        echo "Launching ${casename} on GPU ${gpu_id}"

        CUDA_VISIBLE_DEVICES=${gpu_id} python train_admm_quant.py \
            -s "${DATA_ROOT}/3dovs/${casename}" \
            -m "${MODEL_ROOT}/${casename}" \
            --start_checkpoint "${SEMANTIC_ROOT}/${casename}/chkpnt30000.pth" \
            --port "630${i}" \
            --include_feature \
            > "${LOG_ROOT}/train_admm_quant/${casename}_nohup.log" 2>&1

        CUDA_VISIBLE_DEVICES=${gpu_id} python render_admm_quant.py \
            -s "${DATA_ROOT}/3dovs/${casename}" \
            -m "${MODEL_ROOT}/${casename}" \
            --dataset 3dovs \
            --include_feature \
            > "${LOG_ROOT}/render_admm_quant/${casename}_nohup.log" 2>&1

        CUDA_VISIBLE_DEVICES=${gpu_id} bash "${REPO_ROOT}/eval/eval_3dovs_admm_quant.sh" "${casename}" \
            > "${LOG_ROOT}/eval_admm_quant/${casename}_nohup.log" 2>&1
    } &
done

wait
echo "completed all casenames"
