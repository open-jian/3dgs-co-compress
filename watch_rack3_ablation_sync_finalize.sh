#!/usr/bin/env bash
set -euo pipefail

# Wait for one rack3 result, sync only evaluation-relevant files, then launch a
# local compact fresh-reload evaluation.  `.complete` is transferred last so a
# local watcher can never observe a partially copied checkpoint.
#
# Usage: watch_rack3_ablation_sync_finalize.sh METHOD SCENE GPU
# METHOD: full_joint, stop_grad, sequential, sequential_semantic

METHOD="${1:?METHOD is required}"
SCENE_NAME="${2:?SCENE is required}"
GPU_ID="${3:?GPU is required}"

REMOTE_HOST="${REMOTE_HOST:-r3}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/data2/jian/outputs/wacv27_experiments}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATUS_ROOT="${EXPERIMENT_ROOT}/logs/rack3_sync_finalize/${METHOD}"
STATUS_PATH="${STATUS_ROOT}/${SCENE_NAME}.status"
mkdir -p "${STATUS_ROOT}"
echo "RUNNING" > "${STATUS_PATH}"

case "${METHOD}" in
    full_joint)
        REMOTE_MODEL_ROOT="${EXPERIMENT_ROOT}/clunogs_mvp/lerf_ovs/${SCENE_NAME}/joint_semantic"
        LOCAL_MODEL_ROOT="${REMOTE_MODEL_ROOT}"
        ITERATION=1000
        ;;
    stop_grad)
        REMOTE_MODEL_ROOT="${EXPERIMENT_ROOT}/core_ablations/lerf_ovs/${SCENE_NAME}/joint_stop_semantic_support_grad"
        LOCAL_MODEL_ROOT="${REMOTE_MODEL_ROOT}"
        ITERATION=1000
        ;;
    sequential)
        REMOTE_PIPELINE_ROOT="${EXPERIMENT_ROOT}/clunogs_sequential/lerf_ovs/${SCENE_NAME}"
        REMOTE_MODEL_ROOT="${REMOTE_PIPELINE_ROOT}/semantic_vq"
        LOCAL_PIPELINE_ROOT="${REMOTE_PIPELINE_ROOT}"
        LOCAL_MODEL_ROOT="${LOCAL_PIPELINE_ROOT}/semantic_vq"
        ITERATION=500
        ;;
    sequential_semantic)
        REMOTE_PIPELINE_ROOT="${EXPERIMENT_ROOT}/clunogs_sequential_semantic/lerf_ovs/${SCENE_NAME}"
        REMOTE_MODEL_ROOT="${REMOTE_PIPELINE_ROOT}/semantic_vq"
        LOCAL_PIPELINE_ROOT="${REMOTE_PIPELINE_ROOT}"
        LOCAL_MODEL_ROOT="${LOCAL_PIPELINE_ROOT}/semantic_vq"
        ITERATION=500
        ;;
    *)
        echo "unknown METHOD: ${METHOD}" >&2
        exit 2
        ;;
esac

REMOTE_STATUS="${EXPERIMENT_ROOT}/logs/core_ablations/${METHOD}/${SCENE_NAME}.status"
while true; do
    CURRENT_STATUS="$(ssh -o BatchMode=yes "${REMOTE_HOST}" "cat '${REMOTE_STATUS}' 2>/dev/null" || true)"
    if [[ "${CURRENT_STATUS}" =~ ^[0-9]+$ ]]; then
        break
    fi
    sleep 20
done

if (( CURRENT_STATUS != 0 )); then
    echo "remote_${METHOD}_status=${CURRENT_STATUS}" > "${STATUS_PATH}"
    exit "${CURRENT_STATUS}"
fi

REMOTE_CHECKPOINT="${REMOTE_MODEL_ROOT}/chkpnt${ITERATION}.pth"
REMOTE_QUANTIZATION="${REMOTE_MODEL_ROOT}/sh_quantization_${ITERATION}.pth"
ssh -o BatchMode=yes "${REMOTE_HOST}" \
    "test -f '${REMOTE_CHECKPOINT}' && test -e '${REMOTE_QUANTIZATION}' && test -f '${REMOTE_MODEL_ROOT}/.complete'"

mkdir -p "${LOCAL_MODEL_ROOT}"
rsync -a \
    --exclude='.complete' \
    --exclude='point_cloud/' \
    --exclude='events.out.tfevents.*' \
    --exclude='cameras.json' \
    --exclude='input.ply' \
    "${REMOTE_HOST}:${REMOTE_MODEL_ROOT}/" "${LOCAL_MODEL_ROOT}/"

if [[ "${METHOD}" == "sequential" || "${METHOD}" == "sequential_semantic" ]]; then
    # Follow the remote absolute symlink and materialize the tiny final SH
    # assignment next to the final semantic checkpoint.
    rsync -aL \
        "${REMOTE_HOST}:${REMOTE_QUANTIZATION}" \
        "${LOCAL_MODEL_ROOT}/sh_quantization_${ITERATION}.pth"
    rsync -a \
        "${REMOTE_HOST}:${REMOTE_PIPELINE_ROOT}/budget.txt" \
        "${LOCAL_PIPELINE_ROOT}/budget.txt"
fi

# Transfer the completion marker only after every payload has closed locally.
rsync -a \
    "${REMOTE_HOST}:${REMOTE_MODEL_ROOT}/.complete" \
    "${LOCAL_MODEL_ROOT}/.complete"

mkdir -p \
    "${EXPERIMENT_ROOT}/logs/core_ablations/${METHOD}" \
    "${EXPERIMENT_ROOT}/provenance/core_ablations/${METHOD}"
rsync -a \
    "${REMOTE_HOST}:${EXPERIMENT_ROOT}/logs/core_ablations/${METHOD}/${SCENE_NAME}.log" \
    "${REMOTE_HOST}:${EXPERIMENT_ROOT}/logs/core_ablations/${METHOD}/${SCENE_NAME}.status" \
    "${EXPERIMENT_ROOT}/logs/core_ablations/${METHOD}/"
rsync -a \
    "${REMOTE_HOST}:${EXPERIMENT_ROOT}/provenance/core_ablations/${METHOD}/${SCENE_NAME}.txt" \
    "${EXPERIMENT_ROOT}/provenance/core_ablations/${METHOD}/"

if [[ "${METHOD}" == "full_joint" ]]; then
    # The canonical local watcher was started before training and is already
    # assigned a GPU.  Publishing `.complete` above starts its fresh reload.
    echo "0" > "${STATUS_PATH}"
    exit 0
fi

# Several rack3 jobs can finish together.  Serialize only this watcher's local
# evaluations that share a physical GPU so two fresh-reload processes cannot
# race each other for memory.
GPU_LOCK_ROOT="${EXPERIMENT_ROOT}/logs/rack3_sync_finalize/gpu_locks"
mkdir -p "${GPU_LOCK_ROOT}"
exec 9> "${GPU_LOCK_ROOT}/gpu${GPU_ID}.lock"
flock 9

set +e
/usr/bin/time -v "${REPO_ROOT}/finalize_lerf_model_root.sh" \
    "${SCENE_NAME}" "${LOCAL_MODEL_ROOT}" "${GPU_ID}" "${ITERATION}" \
    > "${STATUS_ROOT}/${SCENE_NAME}.log" 2>&1
FINALIZE_STATUS=$?
set -e
echo "${FINALIZE_STATUS}" > "${STATUS_PATH}"
exit "${FINALIZE_STATUS}"
