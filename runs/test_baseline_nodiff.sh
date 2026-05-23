base_dir="./diffiml_log/diffiml_nodiff"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=8 \
./IMDLBenCo/training_scripts/test.py \
    --model DiffIML_NoDiff \
    --edge_mask_width 7 \
    --infer_time 5 \
    --num_inference_steps 10 \
    --world_size 1 \
    --test_data_json "./runs/dgm_datasets.json" \
    --checkpoint_path "/home/yunfei/IMDLBenCo/log/train_baseline_nodiff" \
    --test_batch_size 16 \
    --image_size 512 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1>${base_dir}/logs.logs