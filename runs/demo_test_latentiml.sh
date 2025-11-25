base_dir="./new_log/noise_log/inf_segfor_cmp_diffiml"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=4 \
./IMDLBenCo/training_scripts/test.py \
    --model LatentIML \
    --edge_mask_width 7 \
    --world_size 1 \
    --test_data_json "./runs/test_datasets.json" \
    --checkpoint_path "/mnt/data0/dubo/workspace/IMDLBenCo/new_log/noise_log/segfor_cmp_diffiml" \
    --test_batch_size 4 \
    --image_size 512 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1>${base_dir}/logs.log