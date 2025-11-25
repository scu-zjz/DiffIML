base_dir="/mnt/data0/dubo/workspace/IMDLBenCo/sota_dir/objectformer_dir"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=4 \
./IMDLBenCo/training_scripts/test.py \
    --model ObjectFormer \
    --edge_mask_width 7 \
    --world_size 1 \
    --test_data_json "./runs/dgm_datasets.json" \
    --checkpoint_path "/mnt/data0/dubo/workspace/IMDLBenCo/sota_dir/objectformer_dir" \
    --test_batch_size 3 \
    --image_size 224 \
    --if_padding \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1>${base_dir}/logs.log