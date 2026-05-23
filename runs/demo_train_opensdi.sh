base_dir="./output_dir_opensdi"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=8 \
./IMDLBenCo/training_scripts/train.py \
    --model MaskCLIP \
    --world_size 1 \
    --batch_size 8 \
    --data_path /mnt/data0/public_datasets/IML/CASIA2.0 \
    --epochs 200 \
    --lr 1e-4 \
    --image_size 512 \
    --if_not_amp \
    --find_unused_parameters \
    --no_model_eval \
    --if_resizing \
    --min_lr 1e-6 \
    --weight_decay 0.05 \
    --edge_mask_width 7 \
    --test_data_path "/mnt/data0/public_datasets/IML/CASIA1.0" \
    --warmup_epochs 1 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 1 \
    --seed 42 \
    --test_period 1 \
    --if_predict_label \
    --mae_pretrain_path "/home/yunfei/mae_pretrain_vit_base.pth" \
2> ${base_dir}/error.log 1>${base_dir}/logs.log