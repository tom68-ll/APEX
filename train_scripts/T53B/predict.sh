#!/bin/bash

devices=0

CUDA_VISIBLE_DEVICES=$devices python -u ./predict_main.py \
--task_path ./data/combine/combine1_perm_1_slightly_processed/task_{}/{} \
--plm_model ./model/t5-v1_1-xl \
--encode_model ./model/t5-base-lm-adapt \
--baseline_name amtm_T53B \
--memory_size 15 \
--frozen_mode tuning_plm \
--max_seq_length 300 \
--encoder_dim 512 \
--T5_patience 10 \
--adapter_patience 30 \
--seed 23 \
--lr 1e-4 \
--adapter_lr 1e-3 \
--epoch 150 \
--adapter_epoch 300 \
--T5_eval_epoch 70 \
--adapter_eval_epoch 50 \
--task_num 7 \
--pool_lambda 0.3 \
--initialize_pool_size 1 \
--initialize_pool_method random \
--key_initialize_method text \
--beam_size 5 \
--batch_size 12 \
--accumulation_step 1 \
--combine_K 6 \
--column_pointer \
--few_shot -1 \
--device $devices \
--max_generate_length 256 \
--model_load_path ./model_saved/T53B/20240929-1924/model/best_model.bin \
--adapter_root_path ./model_saved/T53B/20241024-2218 \
--prediction_save_path ./result/20240929/T53B/T53B_sft \
--load_test_trigger ideal_all_load_ver2 \
--top_n 4 \
--seperate_number 1.6