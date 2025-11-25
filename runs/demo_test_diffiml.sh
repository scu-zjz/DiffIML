base_dir="./diffiml_log/allbench"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=4 \
./IMDLBenCo/training_scripts/test.py \
    --model DiffIML \
    --edge_mask_width 7 \
    --infer_time 5 \
    --num_inference_steps 10 \
    --world_size 1 \
    --test_data_json "./runs/test_datasets.json" \
    --checkpoint_path "/mnt/data0/yunfei/workspace/IMDLBenCo/log/ckpt_dir" \
    --test_batch_size 16 \
    --image_size 512 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1>${base_dir}/logs.log