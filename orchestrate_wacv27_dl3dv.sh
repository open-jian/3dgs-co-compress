#!/usr/bin/env bash
set -u

SECTION6_ROOT=/data2/jian/outputs/wacv27_section6_20260819
SECTION6_RETRY_MARKER=${SECTION6_ROOT}/.teatime_no_sem_retry_finished
DATA_ROOT=/data2/jian/data/dl3dv_wacv27/subset32
DATA_READY=/data2/jian/data/dl3dv_wacv27/.undistortion_complete
OUTPUT_ROOT=/data2/jian/outputs/wacv27_dl3dv_20260819
RGB_CODE=/data2/jian/WACV27_code/gaussian-splatting
LANG_CODE=/data2/jian/WACV27_code/LangSplatV2_joint3scale
COMP_CODE=/data2/jian/WACV27_code/ClunoGS
PYTHON=/home/jian/miniconda3/envs/langsplat_v2/bin/python
RGB_PYTHON=/home/jian/miniconda3/envs/3dgs/bin/python
SAM=/data2/jian/WACV27_code/_shared_checkpoints/sam_vit_h_4b8939.pth
SCENES=(supermarket furniture_store museum business_center)
mkdir -p "${OUTPUT_ROOT}/logs"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

echo "[$(date -Is)] waiting for Section 6 GPU queue"
while [[ ! -f "${SECTION6_ROOT}/.orchestration_complete" ]]; do sleep 30; done
while [[ ! -f "${SECTION6_RETRY_MARKER}" ]]; do sleep 30; done
while [[ ! -f "${DATA_READY}" ]]; do sleep 30; done
echo "[$(date -Is)] starting DL3DV RGB and language-feature construction"

phase_one_pids=()
for index in 0 1 2 3; do
    scene=${SCENES[$index]}
    dataset=${DATA_ROOT}/${scene}
    rgb=${OUTPUT_ROOT}/${scene}/rgb_host
    feature_marker=${OUTPUT_ROOT}/${scene}/.language_features_complete
    mkdir -p "${OUTPUT_ROOT}/${scene}"
    (
        if [[ ! -f "${rgb}/.complete" ]]; then
            cd "${RGB_CODE}" || exit 1
            CUDA_VISIBLE_DEVICES="${index}" /usr/bin/time -v "${RGB_PYTHON}" train.py \
                -s "${dataset}" \
                -m "${rgb}" \
                --eval \
                --iterations 30000 \
                --test_iterations 30000 \
                --save_iterations 30000 \
                --checkpoint_iterations 30000 \
                --port "$((55600 + index))" &&
            touch "${rgb}/.complete"
        fi
    ) > "${OUTPUT_ROOT}/logs/rgb_${scene}.log" 2>&1 &
    phase_one_pids+=("$!")

    (
        if [[ ! -f "${feature_marker}" ]]; then
            cd "${LANG_CODE}" || exit 1
            CUDA_VISIBLE_DEVICES="$((index + 4))" /usr/bin/time -v \
                "${PYTHON}" preprocess.py \
                --dataset_path "${dataset}" \
                --resolution 960 \
                --sam_ckpt_path "${SAM}" &&
            [[ "$(find "${dataset}/language_features" -type f | wc -l)" -eq 64 ]] &&
            touch "${feature_marker}"
        fi
    ) > "${OUTPUT_ROOT}/logs/features_${scene}.log" 2>&1 &
    phase_one_pids+=("$!")
done
for pid in "${phase_one_pids[@]}"; do wait "${pid}" || true; done
echo "[$(date -Is)] DL3DV phase one finished"

# Build one common three-scale LangSplatV2 host per scene.
semantic_pids=()
for index in 0 1 2 3; do
    scene=${SCENES[$index]}
    dataset=${DATA_ROOT}/${scene}
    rgb=${OUTPUT_ROOT}/${scene}/rgb_host
    semantic=${OUTPUT_ROOT}/${scene}/semantic_host
    if [[ ! -f "${rgb}/.complete" || ! -f "${OUTPUT_ROOT}/${scene}/.language_features_complete" ]]; then
        echo "[$(date -Is)] skipping ${scene}: incomplete RGB/features"
        continue
    fi
    (
        if [[ ! -f "${semantic}/.complete" ]]; then
            cd "${LANG_CODE}" || exit 1
            CUDA_VISIBLE_DEVICES="${index}" /usr/bin/time -v "${PYTHON}" train.py \
                -s "${dataset}" \
                -m "${semantic}" \
                --start_checkpoint "${rgb}/chkpnt30000.pth" \
                --feature_level -1 \
                --semantic_level_num 3 \
                --vq_layer_num 1 \
                --codebook_size 64 \
                --cos_loss \
                --topk 4 \
                --iterations 1000 \
                --checkpoint_iterations 1000 \
                --port "$((55700 + index))" &&
            touch "${semantic}/.complete"
        fi
    ) > "${OUTPUT_ROOT}/logs/semantic_host_${scene}.log" 2>&1 &
    semantic_pids+=("$!")
done
for pid in "${semantic_pids[@]}"; do wait "${pid}" || true; done
echo "[$(date -Is)] DL3DV semantic hosts finished"

# Apply the full joint compression schedule.
compression_pids=()
for index in 0 1 2 3; do
    scene=${SCENES[$index]}
    dataset=${DATA_ROOT}/${scene}
    semantic=${OUTPUT_ROOT}/${scene}/semantic_host
    compact=${OUTPUT_ROOT}/${scene}/clunogs
    [[ -f "${semantic}/.complete" ]] || continue
    (
        if [[ ! -f "${compact}/.complete" ]]; then
            cd "${COMP_CODE}" || exit 1
            CUDA_VISIBLE_DEVICES="${index}" /usr/bin/time -v "${PYTHON}" train_joint.py \
                -s "${dataset}" \
                -m "${compact}" \
                --start_checkpoint "${semantic}/chkpnt1000.pth" \
                --reset_iteration \
                --feature_levels 1 2 3 \
                --vq_layer_num 1 \
                --codebook_size 64 \
                --iterations 1000 \
                --semantic_loss cos \
                --normalize \
                --topk 4 \
                --rgb_loss_coeff 1.0 \
                --language_loss_coeff 0.0001 \
                --admm_loss_coeff 1.0 \
                --admm_start_iter 50 \
                --admm_end_iter 950 \
                --admm_interval 450 \
                --simp_iteration1 25 \
                --simp_iteration2 950 \
                --pruning_fraction1 0.00001 \
                --pruning_fraction2 0.5 \
                --rho_opacity 0.0005 \
                --rho_sh 0.0005 \
                --freeze_sh_codebook_iter 900 \
                --sh_codebook_size 256 \
                --port "$((55800 + index))" &&
            touch "${compact}/.complete"
        fi
    ) > "${OUTPUT_ROOT}/logs/compress_${scene}.log" 2>&1 &
    compression_pids+=("$!")
done
for pid in "${compression_pids[@]}"; do wait "${pid}" || true; done
echo "[$(date -Is)] DL3DV compression finished"

# Export and evaluate host/compact artifacts from fresh processes.  DL3DV has
# no LERF-style pixel labels, so this phase intentionally reports RGB only.
evaluation_pids=()
for index in 0 1 2 3; do
    scene=${SCENES[$index]}
    dataset=${DATA_ROOT}/${scene}
    semantic=${OUTPUT_ROOT}/${scene}/semantic_host
    compact=${OUTPUT_ROOT}/${scene}/clunogs
    (
        cd "${COMP_CODE}" || exit 1
        for variant in host ours; do
            if [[ ${variant} == host ]]; then
                model=${semantic}
                checkpoint=${semantic}/chkpnt1000.pth
                sh_args=()
            else
                model=${compact}
                checkpoint=${compact}/chkpnt1000.pth
                sh_args=(--sh-quantization "${compact}/sh_quantization_1000.pth")
            fi
            [[ -f "${checkpoint}" ]] || continue
            deploy=${model}/deploy
            artifact=${deploy}/${scene}.${variant}.pth
            mkdir -p "${deploy}" "${model}/evaluation"
            if [[ ! -f "${artifact}" ]]; then
                "${PYTHON}" compact_artifact.py export \
                    --checkpoint "${checkpoint}" \
                    "${sh_args[@]}" \
                    --output "${artifact}" \
                    --topk 4
            fi
            "${PYTHON}" compact_artifact.py validate --artifact "${artifact}" \
                > "${deploy}/${scene}.${variant}.validation.json"
            if [[ ! -f "${model}/evaluation/rgb.json" ]]; then
                CUDA_VISIBLE_DEVICES="${index}" "${PYTHON}" eval_compact_rgb.py \
                    -s "${dataset}" \
                    -m "${deploy}" \
                    --eval \
                    --compact_artifact "${artifact}" \
                    --output "${model}/evaluation/rgb.json"
            fi
        done
        if [[ -f "${semantic}/evaluation/rgb.json" && -f "${compact}/evaluation/rgb.json" ]]; then
            touch "${OUTPUT_ROOT}/${scene}/.finalized"
        fi
    ) > "${OUTPUT_ROOT}/logs/finalize_${scene}.log" 2>&1 &
    evaluation_pids+=("$!")
done
for pid in "${evaluation_pids[@]}"; do wait "${pid}" || true; done

echo "[$(date -Is)] DL3DV queue finished"
touch "${OUTPUT_ROOT}/.orchestration_complete"
