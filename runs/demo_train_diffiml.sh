#!/usr/bin/env bash
# ===============================================
# DiffIML — Stage 2 Training (Diffusion + LightVAE)
# ===============================================
# Before running:
#   1. Edit runs/balanced_dataset.json with your local dataset paths
#   2. Train LightVAE first via runs/demo_train_light_vae.sh
#   3. Update --light_vae_weights / --test_data_path below
#   4. Adjust GPU IDs and --nproc_per_node to match your machine
# -----------------------------------------------

set -e

base_dir="./log/train_diffiml"
mkdir -p ${base_dir}

# ------- Dataset / weights paths (EDIT ME) -------
DATA_PATH="./runs/balanced_dataset.json"
TEST_DATA_PATH="/path/to/CASIA1.0"
LIGHT_VAE_WEIGHTS="./log/train_light_vae/checkpoints/light_vae_weights.pth"

# ------- GPU / DDP config -------
GPU_IDS="0,1,2,3,4,5,6,7"
OLD_IFS=$IFS; IFS=','; gpus=($GPU_IDS); IFS=$OLD_IFS
NPROC=${#gpus[@]}

CUDA_VISIBLE_DEVICES=${GPU_IDS} \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=${NPROC} \
IMDLBenCo/training_scripts/train.py \
    --model DiffIML \
    --world_size 1 \
    --batch_size 16 \
    --data_path ${DATA_PATH} \
    --test_data_path "${TEST_DATA_PATH}" \
    --light_vae_weights "${LIGHT_VAE_WEIGHTS}" \
    --backbone 'segformer_b3' \
    --pretrain True \
    --prior_rate 0.1 \
    --seg_weight 0.2 \
    --infer_time 5 \
    --num_inference_steps 8 \
    --epochs 60 \
    --lr 1e-4 \
    --min_lr 0 \
    --weight_decay 0.05 \
    --warmup_epochs 2 \
    --image_size 512 \
    --if_resizing \
    --edge_mask_width 7 \
    --save_epochs "5,10,15,20,25,30,35,40,45,50,55" \
    --num_workers 4 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 1 \
    --seed 42 \
    --test_period 1 \
2> ${base_dir}/error.log 1> ${base_dir}/logs.log
