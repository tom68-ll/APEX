# -*- coding: utf-8 -*-
# !/usr/bin/python

import sys
import time
import torch
sys.path.append("..")
import random
from amtm_test_T53B.adapter_trainer_clloss import BasicTrainer
import os
import time
from tqdm import tqdm
import json
import shutil
from sklearn.cluster import KMeans
import torch.nn as nn

sys.path.append('../train_utils')

class AMTMTrainer_T53B(BasicTrainer):
    def __init__(self, args, model_save_path):
        super(AMTMTrainer_T53B, self).__init__(args, model_save_path)

        self.past_task_id = -1
        self.observed_task_ids = []
        self.memory_data = {}  # stores exemplars class by class

        self.best_acc = 0
        self.best_epoch = 0
        self.T5_patience = self.args.T5_patience
        self.adapter_patience = self.args.adapter_patience
        self.T5_eval_epoch = self.args.T5_eval_epoch
        self.adapter_eval_epoch = self.args.adapter_eval_epoch

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
        time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
        self.root_adapter_path = os.path.join(self.root_adapter_path,time_now)
        if not os.path.exists(self.root_adapter_path):
            os.makedirs(self.root_adapter_path)
            print("adapter will be saved at  " + self.root_adapter_path)
        
        self.model_save_path = os.path.join(self.root_adapter_path,"model")
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
            print("model will be saved at  " + self.model_save_path)
        
        continue_train_id = 0
        for task_id in range(continue_train_id, self.args.task_num):

            examples = self.task_controller.task_list[task_id]["train"]
            dev_examples = self.task_controller.task_list[task_id]["dev"]
            n_epochs = self.args.epoch
            train_adapter_epoch = self.args.adapter_epoch

            self.best_acc = 0
            self.best_epoch = 0
            patience = 0
            adapter_skeleton = "adapter_key"

            if task_id == continue_train_id and self.args.model_load_path != "":
                self.model.load_state_dict(torch.load(open(self.args.model_load_path, "rb"), map_location=self.device)["model"])
                dev_acc, _, (_, _, _), _ = self.batch_epoch_acc(self.task_controller.task_list[task_id]["dev"],batch_size=8)
                print("****Loading model from: {} ; dev acc: {}".format(self.args.model_load_path,dev_acc))

                destination_folder = os.path.join(self.root_adapter_path,adapter_skeleton + "_initiate")
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)

                source_folder = self.args.model_load_path.split("/model/")[0] + "/adapter_key_initiate"
                for filename in os.listdir(source_folder):
                    source_path = os.path.join(source_folder, filename)
                    destination_path = os.path.join(destination_folder, filename)
                    
                    shutil.copy(source_path, destination_path)
                    print(f"moved file: {filename}")

            print("********************now taining on {}********************".format(task_id))

            dev_acc_dict = {}
            patience = 0
            
            self.best_acc = 0
            self.best_epoch = 0

            def init_keys():
                self.encoder_model.eval()
                # Determine the number of keys based on the task ID
                #key_num = 10 if task_id == 0 else 5
                key_num = self.args.initialize_pool_size

                # Embed all samples in the dataset
                all_embeddings = []
                data_samples = self.task_controller.task_list[task_id]["train"]
                for sample in data_samples:
                    if self.args.key_initialize_method == "question":
                        text = sample["text"].split('|')[1]
                    elif self.args.key_initialize_method == "text":
                        text = sample["text"]
                    else:
                        raise NotImplementedError("Undefined key initialize method: {}".format(self.args.key_initialize_method))
                    inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        encoder_outputs = self.encoder_model.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                        # Perform mean pooling across the sequence dimension
                        mean_embedding = encoder_outputs.last_hidden_state.mean(dim=1)
                        all_embeddings.append(mean_embedding)

                # Convert list of tensors to a single tensor
                all_embeddings_tensor = torch.stack(all_embeddings).squeeze()  # Resulting tensor shape should be [num_samples, num_features]

                kmeans = KMeans(n_clusters=key_num, random_state=0)
                kmeans.fit(all_embeddings_tensor.cpu().numpy())

                # Use cluster centers as keys
                cluster_centers = kmeans.cluster_centers_
                for center in cluster_centers:
                    center_tensor = torch.tensor(center, dtype=torch.float32, device=self.device)
                    task_embedding_param = nn.Parameter(center_tensor.unsqueeze(0))
                    self.task_keys.append(task_embedding_param)
            
            init_keys()
            self.adapter_path = os.path.join(self.root_adapter_path,adapter_skeleton + str(task_id))
            adapter_name = adapter_skeleton + str(task_id)
            if task_id == 0:
                self.add_adapter(adapter_name)
            else:
                self.load_adapter(adapter_skeleton + str(task_id - 1), load_as = adapter_name)
            for epoch in tqdm(range(train_adapter_epoch)):
                avg_loss, avg_score = self.train_one_epoch_with_adapter_sft(examples, adapter_name, task_id, optimize_step=True)

                if epoch > self.adapter_eval_epoch:
                    if epoch == self.adapter_eval_epoch + 1:
                        self.save_adapter(adapter_name) 
                        print("saved now")
                        self.save_key(self.task_keys[task_id],adapter_name)
                    if dev_examples:
                        dev_acc, _, (_, _, _), _ = self.batch_epoch_acc(dev_examples,batch_size=8)
                        
                        if dev_acc >= self.best_acc:
                            print("****best_epoch!****")
                            self.best_acc, self.best_epoch = dev_acc, epoch
                            self.save_adapter(adapter_name) 
                            print("saved now")
                            self.save_key(self.task_keys[task_id],adapter_name)
                            patience = 0

                            #here to save the dev acc
                            dev_acc_dict[task_id] = self.best_acc
                        else:
                            patience += 1

                        if patience > self.adapter_patience:
                            patience = 0
                            break
                    else:
                        if epoch == self.adapter_eval_epoch + self.adapter_patience:
                            print("****limit_epoch!****")
                            self.best_acc, self.best_epoch = dev_acc, epoch
                            self.save_adapter(adapter_name) 
                            self.save_key(self.task_keys[task_id],adapter_name)
                            break
                    
                    print("\nTask {}, Epoch Train {}, Loss {}, Key_scores {}".format(task_id, epoch,avg_loss,avg_score))
            self.delete_adapter(adapter_name)
            
            print("****task_{}**** best_acc: {} , best_epoch: {} ".format(task_id, self.best_acc,self.best_epoch))
            '''
            choose_rate_list = self.eval_task_stream(task_id,adapter_skeleton)
            '''

            #here to get the lab data after training
            #train_data_split_accuracy = self.calculate_right_train_example(examples,task_id)
            print("--------Now show the lab data for task {}".format(task_id))
            print("--------dev acc dict: ", dev_acc_dict)
            #print("--------train data split accuracy: ", train_data_split_accuracy)
            '''
            print("--------choose right rate list: ", choose_rate_list)
            '''
        #TODO project_layer
        self.save(self.project_layer, name="project_layer.bin")

        return self.avg_acc_list, self.whole_acc_list, self.bwt_list, self.fwt_list