#!/bin/bash

devices=0

CUDA_VISIBLE_DEVICES=$devices python -u ./train_continual_baseline.py \
--task_path ./data/combine/combine1_perm_1_slightly_processed/task_{}/{} \
--plm_model ./model/t5-base-lm-adapt \
--baseline_name adapterpooling \
--memory_size 15 \
--frozen_mode tuning_plm \
--max_seq_length 300 \
--encoder_dim 512 \
--T5_patience 10 \
--adapter_patience 20 \
--seed 23 \
--cuda \
--lr 1e-4 \
--adapter_lr 1e-3 \
--epoch 150 \
--adapter_epoch 200 \
--T5_eval_epoch 70 \
--adapter_eval_epoch 50 \
--task_num 7 \
--pool_lambda 0.3 \
--initialize_pool_size 5 \
--initialize_pool_method random \
--beam_size 5 \
--batch_size 12 \
--accumulation_step 1 \
--combine_K 6 \
--column_pointer \
--few_shot -1 \
--device $devices \
--max_generate_length 256 \
--pet_type adapter \
--root_adapter_path ./model_saved/adapterpool/lora \
--prediction_save_path ./result/20240830/combine_adapterpooling