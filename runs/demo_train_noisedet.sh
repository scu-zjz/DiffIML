base_dir="./log/train_casiav2full_noisedet_50"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=4,5 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=2 \
./IMDLBenCo/training_scripts/train.py \
    --model NoiseDet \
    --world_size 1 \
    --batch_size 36 \
    --data_path /mnt/data0/public_datasets/IML/CASIA2.0 \
    --step 50 \
    --epochs 100 \
    --find_unused_parameters \
    --lr 1e-4 \
    --image_size 512 \
    --if_resizing \
    --min_lr 0 \
    --weight_decay 0.05 \
    --edge_mask_width 7 \
    --test_data_path "/mnt/data0/public_datasets/IML/CASIA1.0" \
    --warmup_epochs 2 \
    --num_workers 8 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 1 \
    --seed 42 \
    --test_period 4 \
2> ${base_dir}/error.log 1>${base_dir}/logs.log