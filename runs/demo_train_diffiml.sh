base_dir="./log/train_mixed_diffiml"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=4 \
IMDLBenCo/training_scripts/train.py \
    --model DiffIML \
    --world_size 1 \
    --batch_size 36 \
    --data_path /mnt/data0/public_datasets/IML/CASIA2.0 \
    --prior_rate 0.0 \
    --pretrain True \
    --seg_weight 0.2 \
    --infer_time 5 \
    --backbone 'segformer_b3' \
    --num_inference_steps 10 \
    --epochs 200 \
    --lr 1e-4 \
    --image_size 512 \
    --if_resizing \
    --save_epochs "37,45,52,61,70,85,93,105,125,149,175" \
    --min_lr 0 \
    --weight_decay 0.05 \
    --edge_mask_width 7 \
    --test_data_path "/mnt/data0/public_datasets/IML/CASIA1.0" \
    --warmup_epochs 2 \
    --num_workers 16 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 1 \
    --seed 42 \
    --test_period 4 \
2> ${base_dir}/error.log 1>${base_dir}/logs.log