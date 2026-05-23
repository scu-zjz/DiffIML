#!/usr/bin/env bash
# ===============================================
# DiffIML — Evaluation on multiple benchmarks
# ===============================================
# Before running:
#   1. Edit runs/test_datasets.json with your local benchmark paths
#   2. Update CHECKPOINT_PATH to point at your trained directory
#      (the script will pick up checkpoint-*.pth files inside it)
# -----------------------------------------------

set -e

base_dir="./log/test_diffiml"
mkdir -p ${base_dir}

# ------- Paths (EDIT ME) -------
CHECKPOINT_PATH="./log/train_diffiml"
TEST_DATA_JSON="./runs/test_datasets.json"

# ------- GPU / DDP config -------
GPU_IDS="0,1,2,3,4,5,6,7"
OLD_IFS=$IFS; IFS=','; gpus=($GPU_IDS); IFS=$OLD_IFS
NPROC=${#gpus[@]}

CUDA_VISIBLE_DEVICES=${GPU_IDS} \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=${NPROC} \
./IMDLBenCo/training_scripts/test.py \
    --model DiffIML \
    --world_size 1 \
    --test_data_json "${TEST_DATA_JSON}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --test_batch_size 16 \
    --image_size 512 \
    --if_resizing \
    --edge_mask_width 7 \
    --infer_time 5 \
    --num_inference_steps 8 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1> ${base_dir}/logs.log
