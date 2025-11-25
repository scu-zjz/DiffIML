base_dir="./log/train_light_vae"
mkdir -p ${base_dir}

DATA_PATH="/mnt/data0/yunfei/workspace/IMDLBenCo/runs/balanced_dataset.json"
ORIGINAL_VAE_PATH="/mnt/data0/yunfei/workspace/model/diffusion/stable_diff/pretrained/vae"
OUTPUT_DIR="${base_dir}/checkpoints"

# RESUME_PATH="${OUTPUT_DIR}/light_vae_epoch_30.pth"

EPOCHS=30
BATCH_SIZE=28
LR=1e-4
NUM_WORKERS=4
GPU_IDS="0,1,2,3"

LATENT_DIM=4
BASE_CHANNELS=32
NORM_LAYER="BatchNorm"
LATENT_WEIGHT=0.01 # 潜空间损失权重

LAYERS_PER_BLOCK=2
ACTIVATION_FN="relu"

OLD_IFS=$IFS
IFS=','
gpus=($GPU_IDS)
IFS=$OLD_IFS
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