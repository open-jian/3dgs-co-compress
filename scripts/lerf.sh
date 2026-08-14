#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="$(dirname "${REPO_ROOT}")"
WORKSPACE_ROOT="$(dirname "${CODE_ROOT}")"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_ROOT}/Output}"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE_ROOT}/Data}"
RGB_ROOT="${OUTPUT_ROOT}/rgb_3dgs/lerf_ovs"
MODEL_ROOT="${OUTPUT_ROOT}/colasplat/semantic/lerf_ovs"
AE_CKPT_ROOT="${OUTPUT_ROOT}/colasplat/autoencoder/ckpt"
LOG_ROOT="${OUTPUT_ROOT}/colasplat/logs/nohup_logs"
export OUTPUT_ROOT DATA_ROOT

casenames=("figurines" "ramen" "teatime" "waldo_kitchen")
gpu_pool=(0 1 2 3 4)
mkdir -p "${LOG_ROOT}/train_ae" "${LOG_ROOT}/render_ae" \
    "${LOG_ROOT}/train" "${LOG_ROOT}/render" "${LOG_ROOT}/eval"
cd "${REPO_ROOT}"

for i in "${!casenames[@]}"; do
    casename=${casenames[$i]}
    gpu_id=${gpu_pool[$i]}
    echo "Launching ${casename} on GPU ${gpu_id}"

    (
        cd autoencoder
        CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
            --dataset_path "${DATA_ROOT}/lerf_ovs/${casename}" \
            --dataset_name "${casename}" \
            --checkpoint_root "${AE_CKPT_ROOT}" \
            --encoder_dims 256 128 64 32 3 \
            --decoder_dims 16 32 64 128 256 256 512 \
            --lr 0.0007 \
            > "${LOG_ROOT}/train_ae/${casename}.log" 2>&1

        CUDA_VISIBLE_DEVICES=${gpu_id} python test.py \
            --dataset_path "${DATA_ROOT}/lerf_ovs/${casename}" \
            --dataset_name "${casename}" \
            --checkpoint_root "${AE_CKPT_ROOT}" \
            > "${LOG_ROOT}/render_ae/${casename}.log" 2>&1
        cd "${REPO_ROOT}"

        CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
            -s "${DATA_ROOT}/lerf_ovs/${casename}" \
            -m "${MODEL_ROOT}/${casename}" \
            --start_checkpoint "${RGB_ROOT}/${casename}/chkpnt30000.pth" \
            --port "700${i}" \
            > "${LOG_ROOT}/train/${casename}.log" 2>&1

        CUDA_VISIBLE_DEVICES=${gpu_id} python render.py \
            -s "${DATA_ROOT}/lerf_ovs/${casename}" \
            -m "${MODEL_ROOT}/${casename}" \
            --include_feature \
            > "${LOG_ROOT}/render/${casename}.log" 2>&1

        CUDA_VISIBLE_DEVICES=${gpu_id} bash "${REPO_ROOT}/eval/eval_lerf.sh" "${casename}" \
            > "${LOG_ROOT}/eval/${casename}.log" 2>&1
    ) &
done

wait
echo "finished all casenames"
