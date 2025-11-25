base_dir="./new_log/noise_log/segfor_cmp_diffiml"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=4 \
./IMDLBenCo/training_scripts/train.py \
    --model LatentIML \
    --world_size 1 \
    --batch_size 44 \
    --data_path '/mnt/data0/public_datasets/IML/CASIAv2_full.json' \
    --epochs 100 \
    --lr 1e-4 \
    --image_size 512 \
    --find_unused_parameters \
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