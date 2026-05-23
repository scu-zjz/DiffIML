base_dir="./diffiml_log/eval_dir_opensdi"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=8 \
./IMDLBenCo/training_scripts/test.py \
    --model MaskCLIP \
    --edge_mask_width 7 \
    --world_size 1 \
    --test_data_json "./runs/temp.json" \
    --checkpoint_path "./output_dir_opensdi" \
    --test_batch_size 16 \
    --image_size 512 \
    --no_model_eval \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --if_predict_label \
    --mae_pretrain_path "/home/yunfei/mae_pretrain_vit_base.pth" \
2> ${base_dir}/error.log 1>${base_dir}/logs.log