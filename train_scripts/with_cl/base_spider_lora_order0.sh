#!/bin/bash

devices=0

CUDA_VISIBLE_DEVICES=$devices python -u ./train_continual_baseline.py \
--task_path ./data/spider/spider_perm_1_slightly_processed/task_{}/{} \
--plm_model ./model/t5-base-lm-adapt \
--baseline_name amtm_clloss \
--memory_size 15 \
--frozen_mode tuning_plm \
--max_seq_length 300 \
--encoder_dim 512 \
--T5_patience 10 \
--adapter_patience 30 \
--seed 23 \
--cuda \
--lr 1e-4 \
--adapter_lr 1e-3 \
--epoch 150 \
--adapter_epoch 300 \
--T5_eval_epoch 50 \
--adapter_eval_epoch 50 \
--task_num 11 \
--pool_lambda 0.3 \
--initialize_pool_size 5 \
--initialize_pool_method random \
--key_initialize_method text \
--beam_size 5 \
--batch_size 8 \
--accumulation_step 1 \
--combine_K 6 \
--column_pointer \
--few_shot -1 \
--device $devices \
--max_generate_length 256 \
--pet_type lora \
--model_load_path ./model_saved/withcl/20240904-2349/model/best_model.bin \
--root_adapter_path ./model_saved/withcl \
--prediction_save_path ./result/20241009/T5-base/spider/lora/withcl_order0