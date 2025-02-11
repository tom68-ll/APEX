# -*- coding: utf-8 -*-
# !/usr/bin/python

import sys
import time
import torch
sys.path.append("..")
import random
from load_test_for_T53B.adapter_trainer import BasicTrainer
import os
import time
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import json
from tqdm import tqdm

sys.path.append('../train_utils')


class ProjectLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation=F.relu):
        super(ProjectLayer, self).__init__()
        # W_1: Linear layer from input_dim to hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # W_2: Linear layer from hidden_dim to output_dim
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Activation function σ(·)
        self.activation = activation

    def forward(self, r_i):
        # First apply W_1 and activation function σ(W_1 * r_i)
        hidden = self.activation(self.fc1(r_i))
        # Then apply W_2
        h_i = self.fc2(hidden)
        return h_i



class AMTMTrainer_T53B(BasicTrainer):
    def __init__(self, args, model_save_path):
        super(AMTMTrainer_T53B, self).__init__(args, model_save_path)

        # CL component
        self.past_task_id = -1
        self.observed_task_ids = []
        self.memory_data = {}  # stores exemplars class by class

        self.best_acc = 0
        self.best_epoch = 0
        self.T5_patience = self.args.T5_patience
        self.adapter_patience = self.args.adapter_patience
        self.T5_eval_epoch = self.args.T5_eval_epoch
        self.adapter_eval_epoch = self.args.adapter_eval_epoch

        input_dim = self.encoder_model.config.hidden_size
        hidden_dim = 512
        output_dim = 256
        self.project_layer = ProjectLayer(input_dim, hidden_dim, output_dim).to(self.device)

    def calculate_right_train_example(self,examples_dict : dict, task_id : int) -> float:
        all_num = 0
        right_num = 0
        for key_idx_foreal in list(examples_dict.keys()):
            examples_to_test = examples_dict[key_idx_foreal]
            result_dict = self.make_train_classification(examples_to_test,task_id)
            all_num += len(examples_to_test)
            right_num += len(result_dict[key_idx_foreal])
        return right_num / all_num

    def train(self):
        #change adapter path according to time
        self.root_adapter_path = self.args.adapter_root_path
        adapter_skeleton = "adapter_key"
        key_num = self.args.initialize_pool_size
        self.load_all_keys(self.root_adapter_path,adapter_skeleton,key_num * self.args.task_num)
        for idx in range(len(self.task_keys)):
            self.task_keys[idx].requires_grad = False
        #TODO here is all the code for project_layer
        '''
        if self.args.project_layer_path:
            self.load(self.project_layer,name = "project_layer.bin",path = self.args.project_layer_path)
        tmp_task_keys = []
        for key in self.task_keys:
            tmp_task_keys.append(self.project_layer(key))
        self.task_keys = tmp_task_keys
        for param in self.project_layer.parameters():
            param.requires_grad = False
        
        embeddings_save_path = self.args.prediction_save_path
        original_embeddings_path = os.path.join(embeddings_save_path, "original_embeddings")
        projected_embeddings_path = os.path.join(embeddings_save_path, "projected_embeddings")
        
        os.makedirs(original_embeddings_path, exist_ok=True)
        os.makedirs(projected_embeddings_path, exist_ok=True)
        '''
        first_task_id = 0
        time_start = time.time()
        for task_id in range(first_task_id, self.args.task_num):
            #T5_train
            if task_id == first_task_id and self.args.model_load_path != "":
                self.model.load_state_dict(torch.load(open(self.args.model_load_path, "rb"), map_location=self.device)["model"])
                dev_acc, _, (_, _, _), _ = self.batch_epoch_acc(self.task_controller.task_list[task_id]["dev"],batch_size=8)
                print("****Loading model from: {} ; dev acc: {}".format(self.args.model_load_path,dev_acc))

            print("****task_{}**** best_acc: {} , best_epoch: {} ".format(task_id, self.best_acc,self.best_epoch))
            
            choose_rate_list = self.eval_task_stream(task_id,adapter_skeleton)