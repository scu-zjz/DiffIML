base_dir="./log/iml_vit_catnet"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=8 \
./IMDLBenCo/training_scripts/train.py \
    --model IML_ViT \
    --edge_lambda 20 \
    --vit_pretrain_path /mnt/data0/xiaochen/workspace/IML—ViT_250124/mae_pretrain_vit_base.pth \
    --world_size 1 \
    --batch_size 2 \
    --data_path "/home/yunfei/IMDLBenCo/runs/balanced_dataset.json" \
    --epochs 50 \
    --lr 1e-4 \
    --image_size 1024 \
    --if_resizing \
    --save_epochs "25,35,45" \
    --min_lr 5e-7 \
    --weight_decay 0.05 \
    --edge_mask_width 7 \
    --test_data_path "/mnt/data0/public_datasets/IML/CASIA1.0" \
    --warmup_epochs 2 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 8 \
    --seed 42 \
    --test_period 4 \
    --resume "/home/yunfei/IMDLBenCo/log/iml_vit_catnet/checkpoint-14.pth"
2> ${base_dir}/error.log 1>${base_dir}/logs.log