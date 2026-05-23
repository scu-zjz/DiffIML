#!/usr/bin/env bash
# ===============================================
# DiffIML — Stage 1: Train LightVAE
#   A slim 1-channel VAE distilled from SD-VAE,
#   used to compress mask & edge maps into latents.
# ===============================================
# Before running:
#   1. Edit runs/balanced_dataset.json with your local data paths.
#   2. (Optional) Download SD-VAE pretrained weights and set ORIGINAL_VAE_PATH
#      if you want distillation; otherwise it will fall back to random init.
# -----------------------------------------------

set -e

base_dir="./log/train_light_vae"
mkdir -p ${base_dir}

# ------- Paths (EDIT ME) -------
DATA_PATH="./runs/balanced_dataset.json"
ORIGINAL_VAE_PATH="/path/to/sd-vae-ft-mse"          # optional teacher VAE
OUTPUT_DIR="${base_dir}/checkpoints"

# ------- Training Hyper-parameters -------
EPOCHS=40
BATCH_SIZE=8
LR=1e-4
NUM_WORKERS=4
GPU_IDS="0,1,2,3,4,5,6,7"

LATENT_DIM=4
BASE_CHANNELS=32
NORM_LAYER="BatchNorm"
LATENT_WEIGHT=0.01
LAYERS_PER_BLOCK=2
ACTIVATION_FN="relu"

OLD_IFS=$IFS; IFS=','; gpus=($GPU_IDS); IFS=$OLD_IFS
NPROC=${#gpus[@]}

CUDA_VISIBLE_DEVICES=${GPU_IDS} \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=${NPROC} \
IMDLBenCo/training_scripts/train_light_vae.py \
    --data_path ${DATA_PATH} \
    --original_vae_path ${ORIGINAL_VAE_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --output_filename "light_vae_weights.pth" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --num_workers ${NUM_WORKERS} \
    --image_size 512 \
    --save_interval 10 \
    --latent_dim ${LATENT_DIM} \
    --base_channels ${BASE_CHANNELS} \
    --norm_layer_type ${NORM_LAYER} \
    --latent_loss_weight ${LATENT_WEIGHT} \
    --if_resizing \
    --edge_mask_width 7 \
    --layers_per_block ${LAYERS_PER_BLOCK} \
    --activation_fn ${ACTIVATION_FN} \
2> ${base_dir}/error.log 1> ${base_dir}/logs.log
