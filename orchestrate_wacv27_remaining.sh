#!/usr/bin/env bash
set -u

ROOT=/data2/jian/outputs/wacv27_section6_20260819
CODE=/data2/jian/WACV27_code/ClunoGS
SCENES=(figurines ramen teatime waldo_kitchen)
mkdir -p "${ROOT}/logs"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

echo "[$(date -Is)] waiting for component wave fresh-reload evaluations"
while true; do
    ready=0
    for variant in spars_only vq_only; do
        for scene in "${SCENES[@]}"; do
            [[ -f "${ROOT}/${variant}/${scene}/.finalized" ]] && ready=$((ready + 1))
        done
    done
    [[ ${ready} -eq 8 ]] && break
    sleep 20
done
echo "[$(date -Is)] component wave finalized"

# Wave two, cards 0--3: no-dual-feedback controls.
no_dual_pids=()
for gpu in 0 1 2 3; do
    scene=${SCENES[$gpu]}
    (
        "${CODE}/run_wacv27_control_variant.sh" no_dual "${scene}" "${gpu}" 1000 "$((55300 + gpu))" &&
        "${CODE}/finalize_wacv27_control.sh" \
            "${scene}" "${ROOT}/no_dual/${scene}" "${gpu}" 1000
    ) > "${ROOT}/logs/no_dual_${scene}.log" 2>&1 &
    no_dual_pids+=("$!")
done

# Smoke the reverse sequential implementation before occupying cards 4--7.
reverse_smoke_root=/data2/jian/outputs/wacv27_section6_reverse_smoke_20260819
if CONTROL_OUTPUT_ROOT="${reverse_smoke_root}" \
    "${CODE}/run_wacv27_reverse_sequential.sh" ramen 4 80 55480 \
    > "${ROOT}/logs/vq_then_spars_smoke.log" 2>&1; then
    echo "[$(date -Is)] reverse sequential smoke passed"
    reverse_pids=()
    for index in 0 1 2 3; do
        gpu=$((index + 4))
        scene=${SCENES[$index]}
        (
            "${CODE}/run_wacv27_reverse_sequential.sh" \
                "${scene}" "${gpu}" 1000 "$((55400 + index))" &&
            "${CODE}/finalize_wacv27_control.sh" \
                "${scene}" "${ROOT}/vq_then_spars/${scene}/spars_second" \
                "${gpu}" 500
        ) > "${ROOT}/logs/vq_then_spars_${scene}.log" 2>&1 &
        reverse_pids+=("$!")
    done
else
    echo "[$(date -Is)] reverse sequential smoke failed; skipping four formal runs"
    reverse_pids=()
fi

for pid in "${no_dual_pids[@]}"; do wait "${pid}" || true; done
for pid in "${reverse_pids[@]}"; do wait "${pid}" || true; done
echo "[$(date -Is)] wave two finished"

# Wave three records exact retained host-row IDs for the semantic-retention
# diagnostic.  Individual failures are allowed and do not stop other scenes.
tracked_pids=()
for index in 0 1 2 3; do
    scene=${SCENES[$index]}
    (
        "${CODE}/run_wacv27_control_variant.sh" \
            full_tracked "${scene}" "${index}" 1000 "$((55500 + index))"
    ) > "${ROOT}/logs/full_tracked_${scene}.log" 2>&1 &
    tracked_pids+=("$!")
    gpu=$((index + 4))
    (
        "${CODE}/run_wacv27_control_variant.sh" \
            no_sem_tracked "${scene}" "${gpu}" 1000 "$((55504 + index))"
    ) > "${ROOT}/logs/no_sem_tracked_${scene}.log" 2>&1 &
    tracked_pids+=("$!")
done
for pid in "${tracked_pids[@]}"; do wait "${pid}" || true; done

echo "[$(date -Is)] all scheduled Section 6 training waves finished"
touch "${ROOT}/.orchestration_complete"
