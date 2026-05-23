base_dir="./log/train_baseline_nodiff"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=8 \
IMDLBenCo/training_scripts/train.py \
    --model DiffIML_NoDiff \
    --world_size 1 \
    --batch_size 16 \
    --data_path "/mnt/data0/public_datasets/IML/CASIA2.0" \
    --prior_rate 0.0 \
    --pretrain True \
    --seg_weight 0.2 \
    --infer_time 5 \
    --backbone 'segformer_b3' \
    --num_inference_steps 10 \
    --epochs 150 \
    --lr 1e-4 \
    --image_size 512 \
    --if_resizing \
    --save_epochs "20,35,45,61,70,86,93,125" \
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
    --test_period 1 \
2> ${base_dir}/error.log 1>${base_dir}/logs.log