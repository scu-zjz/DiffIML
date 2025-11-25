for ensemble_num in 1 3 5 10; do
    for i in 1 2 3 4 5; do
        # 根据 ensemble_num 和 i 来设置 base_dir 和 --infer_time
        base_dir="./diffiml_log/ensemble_${ensemble_num}_${i}"
        mkdir -p ${base_dir}

        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        torchrun  \
            --standalone    \
            --nnodes=1     \
            --nproc_per_node=4 \
        ./IMDLBenCo/training_scripts/test.py \
            --model DiffIML \
            --edge_mask_width 7 \
            --infer_time ${ensemble_num} \
            --num_inference_steps 8 \
            --world_size 1 \
            --test_data_json "./runs/casiav1.json" \
            --checkpoint_path "/mnt/data0/yunfei/workspace/IMDLBenCo/log/ckpt_dir" \
            --test_batch_size 64 \
            --image_size 512 \
            --if_resizing \
            --output_dir ${base_dir}/ \
            --log_dir ${base_dir}/ \
        2> ${base_dir}/error.log 1>${base_dir}/logs.log
    done
done
