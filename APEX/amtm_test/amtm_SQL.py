# -*- coding: utf-8 -*-
# !/usr/bin/python

import sys
import time
import torch
sys.path.append("..")
import random
from amtm_test.SQL_trainer import SQLTrainer
import os
import time
from tqdm import tqdm

sys.path.append('../train_utils')


class AMTMSQL(SQLTrainer):
    def __init__(self, args, model_save_path):
        super(AMTMSQL, self).__init__(args, model_save_path)

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
        time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
        self.root_adapter_path = os.path.join(self.root_adapter_path,time_now)
        if not os.path.exists(self.root_adapter_path):
            os.makedirs(self.root_adapter_path)
            print("adapter will be saved at  " + self.root_adapter_path)
        
        self.model_save_path = os.path.join(self.root_adapter_path,"model")
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
            print("model will be saved at  " + self.model_save_path)

        for task_id in range(self.args.task_num):

            examples = self.task_controller.task_list[task_id]["train"]
            dev_examples = self.task_controller.task_list[task_id]["dev"]
            n_epochs = self.args.epoch
            train_adapter_epoch = self.args.adapter_epoch

            self.best_acc = 0
            self.best_epoch = 0
            patience = 0
            adapter_skeleton = "adapter_key"

            if task_id == 0 and self.args.model_load_path == "":
                
                optimizer_grouped_parameters = self.make_optimizer_groups_forT5()
                self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.args.lr)
                print("********************now taining on {}********************".format(task_id))
                for epoch in tqdm(range(n_epochs)):
                    self.model.train()

                    random.shuffle(examples)
                    st = 0
                    report_loss, example_num = 0.0, 0
                    total_scores = 0.0
                    cnt = 0
                    self.optimizer.zero_grad()

                    while st < len(examples):
                        ed = st + self.args.batch_size if st + self.args.batch_size < len(examples) else len(examples)
                        report_loss, example_num, model_loss = self.train_one_batch(examples[st:ed], report_loss, example_num)

                        loss = model_loss
                        loss.backward()

                        if self.args.clip_grad > 0.:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        st = ed
                        cnt += 1

                    #now start early stop
                    if epoch >= self.T5_eval_epoch:
                        dev_acc, _, (_, _, _), _ = self.batch_epoch_acc(self.task_controller.task_list[task_id]["dev"],batch_size=32)
                        if dev_acc >= self.best_acc:
                            print("****best_epoch!****")
                            self.best_acc, self.best_epoch = dev_acc, epoch
                            self.save(self.model, name="best_model.bin")
                            patience = 0
                        else:
                            patience += 1

                        if patience > self.T5_patience:
                            break

                    print("\nTask {}, Epoch Train {}, Loss {}".format(task_id, epoch, report_loss / example_num))
                print("****task_{}**** best_acc: {} , best_epoch: {} ".format(task_id, self.best_acc,self.best_epoch))
                self.load(self.model,"best_model.bin")
            
            if task_id == 0 and self.args.model_load_path != "":
                self.model.load_state_dict(torch.load(open(self.args.model_load_path, "rb"), map_location=self.device)["model"])
                dev_acc, _, (_, _, _), _ = self.batch_epoch_acc(self.task_controller.task_list[task_id]["dev"],batch_size=32)
                print("****Loading model from: {} ; dev acc: {}".format(self.args.model_load_path,dev_acc))

            
            self.initialize_adapter_pool(task_id,adapter_skeleton) 
            examples_dict = self.make_train_classification(examples,task_id)
            dev_examples_dict = self.make_train_classification(dev_examples,task_id)
            for dev_k in list(dev_examples_dict.keys()):
                print(dev_k,len(dev_examples_dict[dev_k]))
            print("********************now taining on {}********************".format(task_id))

            dev_acc_dict = {}

            for key_idx in list(examples_dict.keys()):
                self.best_acc = 0
                self.best_epoch = 0
                
                child_examples = examples_dict[key_idx]
                self.adapter_path = os.path.join(self.root_adapter_path,adapter_skeleton + str(key_idx))
                adapter_name = adapter_skeleton + str(key_idx)
                self.load_adapter(adapter_name)
                dev_child_examples = dev_examples_dict[key_idx] if key_idx in list(dev_examples_dict.keys()) else None
                for epoch in tqdm(range(train_adapter_epoch)):
                    avg_loss, avg_score = self.train_one_epoch_with_adapter(child_examples, adapter_name, key_idx, optimize_step=True)

                    if epoch > self.adapter_eval_epoch:
                        if dev_child_examples:
                            dev_acc, _, (_, _, _), _ = self.batch_epoch_acc(dev_child_examples,batch_size=32)
                            
                            if dev_acc >= self.best_acc:
                                print("****best_epoch!****")
                                self.best_acc, self.best_epoch = dev_acc, epoch
                                self.save_adapter(adapter_name) 
                                patience = 0

                                #here to save the dev acc
                                dev_acc_dict[key_idx] = self.best_acc
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
                                break
                    

                    print("\nTask {}, Epoch Train {}, Loss {}, Key_scores {}".format(task_id, epoch,avg_loss,avg_score))
                self.delete_adapter(adapter_name)
            print("****task_{}**** best_acc: {} , best_epoch: {} ".format(task_id, self.best_acc,self.best_epoch))
            choose_rate_list = self.eval_task_stream(task_id,adapter_skeleton)

            #here to get the lab data after training
            train_data_split_accuracy = self.calculate_right_train_example(examples_dict,task_id)
            print("--------Now show the lab data for task {}".format(task_id))
            print("--------dev acc dict: ", dev_acc_dict)
            print("--------train data split accuracy: ", train_data_split_accuracy)
            print("--------choose right rate list: ", choose_rate_list)


        return self.avg_acc_list, self.whole_acc_list, self.bwt_list, self.fwt_list