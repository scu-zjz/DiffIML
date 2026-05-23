#!/usr/bin/env bash
# ===============================================
# DiffIML — Robustness evaluation
#   Tests under various perturbations:
#   - Gaussian Blur / Noise
#   - JPEG compression
#   - Resize
# ===============================================

set -e

base_dir="./log/robust_diffiml"
mkdir -p ${base_dir}

# ------- Paths (EDIT ME) -------
TEST_DATA_PATH="/path/to/CASIA1.0"
CHECKPOINT_FILE="./log/train_diffiml/checkpoint-best.pth"

GPU_IDS="0,1,2,3,4,5,6,7"
OLD_IFS=$IFS; IFS=','; gpus=($GPU_IDS); IFS=$OLD_IFS
NPROC=${#gpus[@]}

CUDA_VISIBLE_DEVICES=${GPU_IDS} \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=${NPROC} \
./IMDLBenCo/training_scripts/test_robust.py \
    --model DiffIML \
    --edge_mask_width 7 \
    --world_size 1 \
    --test_data_path "${TEST_DATA_PATH}" \
    --checkpoint_path "${CHECKPOINT_FILE}" \
    --test_batch_size 20 \
    --image_size 512 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1> ${base_dir}/logs.log
