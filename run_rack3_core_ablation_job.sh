#!/usr/bin/env bash
set -uo pipefail

# Run one rack3 ablation with durable log/status/provenance files.
# Usage: run_rack3_core_ablation_job.sh METHOD SCENE GPU PORT SOURCE_CKPT
# METHOD is "full_joint", "stop_grad", "sequential", or
# "sequential_semantic".

METHOD="${1:?METHOD is required}"
SCENE_NAME="${2:?SCENE is required}"
GPU_INDEX="${3:?GPU is required}"
PORT="${4:?PORT is required}"
SOURCE_CHECKPOINT="${5:?SOURCE_CKPT is required}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/data2/jian/outputs/wacv27_experiments}"
PYTHON_BIN="${PYTHON_BIN:-/home/jian/.local/share/mamba/envs/langsplat_v2/bin/python}"
LOG_ROOT="${EXPERIMENT_ROOT}/logs/core_ablations/${METHOD}"
PROVENANCE_ROOT="${EXPERIMENT_ROOT}/provenance/core_ablations/${METHOD}"
LOG_PATH="${LOG_ROOT}/${SCENE_NAME}.log"
STATUS_PATH="${LOG_ROOT}/${SCENE_NAME}.status"
PROVENANCE_PATH="${PROVENANCE_ROOT}/${SCENE_NAME}.txt"

case "${METHOD}" in
    full_joint)
        RUNNER="${REPO_ROOT}/run_lerf_scene_full_joint.sh"
        EXPECTED_ROOT="${EXPERIMENT_ROOT}/clunogs_mvp/lerf_ovs/${SCENE_NAME}/joint_semantic"
        EXPECTED_CHECKPOINT="${EXPECTED_ROOT}/chkpnt1000.pth"
        EXPECTED_QUANTIZATION="${EXPECTED_ROOT}/sh_quantization_1000.pth"
        ;;
    stop_grad)
        RUNNER="${REPO_ROOT}/run_lerf_scene_stop_grad.sh"
        EXPECTED_ROOT="${EXPERIMENT_ROOT}/core_ablations/lerf_ovs/${SCENE_NAME}/joint_stop_semantic_support_grad"
        EXPECTED_CHECKPOINT="${EXPECTED_ROOT}/chkpnt1000.pth"
        EXPECTED_QUANTIZATION="${EXPECTED_ROOT}/sh_quantization_1000.pth"
        ;;
    sequential)
        RUNNER="${REPO_ROOT}/run_lerf_scene_sequential.sh"
        EXPECTED_ROOT="${EXPERIMENT_ROOT}/clunogs_sequential/lerf_ovs/${SCENE_NAME}"
        EXPECTED_CHECKPOINT="${EXPECTED_ROOT}/semantic_vq/chkpnt500.pth"
        EXPECTED_QUANTIZATION="${EXPECTED_ROOT}/semantic_vq/sh_quantization_500.pth"
        ;;
    sequential_semantic)
        RUNNER="${REPO_ROOT}/run_lerf_scene_sequential_semantic.sh"
        EXPECTED_ROOT="${EXPERIMENT_ROOT}/clunogs_sequential_semantic/lerf_ovs/${SCENE_NAME}"
        EXPECTED_CHECKPOINT="${EXPECTED_ROOT}/semantic_vq/chkpnt500.pth"
        EXPECTED_QUANTIZATION="${EXPECTED_ROOT}/semantic_vq/sh_quantization_500.pth"
        ;;
    *)
        echo "unknown METHOD: ${METHOD}" >&2
        exit 2
        ;;
esac

mkdir -p "${LOG_ROOT}" "${PROVENANCE_ROOT}"

# Reserve ramen's released GPU for the required 98 GB teatime full-joint run
# before ramen's queued sequential stage starts.
RAMEN_PRIORITY_MARKER="${EXPERIMENT_ROOT}/logs/core_ablations/pipelines/ramen_waits_for_teatime_full"
if [[ "${METHOD}" == "sequential" && "${SCENE_NAME}" == "ramen" && -f "${RAMEN_PRIORITY_MARKER}" ]]; then
    TEATIME_FULL_STATUS="${EXPERIMENT_ROOT}/logs/core_ablations/full_joint/teatime.status"
    while true; do
        DEPENDENCY_STATUS="$(cat "${TEATIME_FULL_STATUS}" 2>/dev/null || true)"
        if [[ "${DEPENDENCY_STATUS}" =~ ^[0-9]+$ ]]; then
            break
        fi
        sleep 20
    done
fi

START_ISO="$(date -Is)"
START_EPOCH="$(date +%s)"

{
    echo "method=${METHOD}"
    echo "scene=${SCENE_NAME}"
    echo "host=$(hostname -f 2>/dev/null || hostname)"
    echo "gpu_index=${GPU_INDEX}"
    echo "gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU_INDEX}" | head -n 1)"
    echo "port=${PORT}"
    echo "start=${START_ISO}"
    echo "source_checkpoint=${SOURCE_CHECKPOINT}"
    stat -c 'source_bytes=%s' "${SOURCE_CHECKPOINT}"
    stat -c 'source_mtime=%y' "${SOURCE_CHECKPOINT}"
    echo "python=${PYTHON_BIN}"
    "${PYTHON_BIN}" -c 'import torch; print("torch=" + torch.__version__); print("torch_cuda=" + str(torch.version.cuda))'
    sha256sum \
        "${REPO_ROOT}/train_joint.py" \
        "${REPO_ROOT}/gaussian_renderer/__init__.py" \
        "${RUNNER}"
    printf 'command=CUDA_VISIBLE_DEVICES=%q PYTHON_BIN=%q EXPERIMENT_ROOT=%q %q %q 1000 %q %q\n' \
        "${GPU_INDEX}" "${PYTHON_BIN}" "${EXPERIMENT_ROOT}" \
        "${RUNNER}" "${SCENE_NAME}" "${PORT}" "${SOURCE_CHECKPOINT}"
} > "${PROVENANCE_PATH}"

if [[ -f "${EXPECTED_ROOT}/.complete" && -f "${EXPECTED_CHECKPOINT}" && -e "${EXPECTED_QUANTIZATION}" ]]; then
    echo "0" > "${STATUS_PATH}"
    {
        echo "reused=true"
        echo "end=$(date -Is)"
        echo "elapsed_seconds=0"
    } >> "${PROVENANCE_PATH}"
    exit 0
fi

echo "RUNNING" > "${STATUS_PATH}"
{
    echo "[$(date -Is)] starting ${METHOD} ${SCENE_NAME} on physical GPU ${GPU_INDEX}"
    CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
        PYTHON_BIN="${PYTHON_BIN}" \
        EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" \
        "${RUNNER}" "${SCENE_NAME}" 1000 "${PORT}" "${SOURCE_CHECKPOINT}"
} > "${LOG_PATH}" 2>&1
STATUS=$?
END_EPOCH="$(date +%s)"
END_ISO="$(date -Is)"
ELAPSED_SECONDS=$(( END_EPOCH - START_EPOCH ))

echo "${STATUS}" > "${STATUS_PATH}"
{
    echo "reused=false"
    echo "end=${END_ISO}"
    echo "elapsed_seconds=${ELAPSED_SECONDS}"
    echo "exit_status=${STATUS}"
    if [[ -f "${EXPECTED_CHECKPOINT}" ]]; then
        stat -c 'checkpoint_bytes=%s' "${EXPECTED_CHECKPOINT}"
        stat -c 'checkpoint_mtime=%y' "${EXPECTED_CHECKPOINT}"
    fi
    if [[ -e "${EXPECTED_QUANTIZATION}" ]]; then
        stat -Lc 'quantization_bytes=%s' "${EXPECTED_QUANTIZATION}"
        stat -Lc 'quantization_mtime=%y' "${EXPECTED_QUANTIZATION}"
    fi
} >> "${PROVENANCE_PATH}"

exit "${STATUS}"
