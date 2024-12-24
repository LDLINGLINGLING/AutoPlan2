formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1 

deepspeed --include localhost:0,1,2,3,4,5,6,7 finetune_autoplan.py \
    --model_name_or_path /root/ld/ld_model_pretrain/Qwen2.5-3B-Instruct \
    --output_dir output/AutoPlan2/$formatted_time/ \
    --train_data_path /root/ld/ld_project/MiniCPM/finetune/data/autoplan2/autoplan2_chatml_train.json \
    --eval_data_path /root/ld/ld_project/MiniCPM/finetune/data/autoplan2/autoplan2_chatml_test.json \
    --learning_rate 4e-5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1  \
    --model_max_length 2200 \
    --bf16  \
    --gradient_accumulation_steps 8 \
    --warmup_steps 100 \
    --max_steps 500 \
    --weight_decay 0.01 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 500 \
    --seed 42 \
    --log_level info \
    --logging_strategy steps \
    --logging_steps 10 \
    --deepspeed configs/ds_config_zero3_offload.json
