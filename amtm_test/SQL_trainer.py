# -*- coding: utf-8 -*-
# !/usr/bin/python

import os
import sys
import torch
from torch import nn
import random
sys.path.append("..")
from rule.define_rule import C
from task_loader import TaskLoader
from rule import define_rule
from train_utils.utils import save_args, calc_beam_acc
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
import sqlite3
import adapters
import random
from tqdm import tqdm
import time
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score


class SQLTrainer(object):

    def __init__(self, args, model_save_path):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.grammar = define_rule.Grammar(is_sketch=None)

        print("Init the Model ...")
        self.model = T5ForConditionalGeneration.from_pretrained(self.args.plm_model).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(self.args.plm_model)
        self.tokenizer.add_tokens(['<','<='])
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.truncation_side = 'left'

        self.encoder_model = T5ForConditionalGeneration.from_pretrained(self.args.plm_model).to(self.device)
        self.encoder_model.resize_token_embeddings(len(self.tokenizer))

        print("Load Task Data ...")
        self.task_controller = TaskLoader(args)

        self.args.task_num = len(self.task_controller.task_list)

        self.first_acc_list = [-1 for i in range(self.args.task_num)]
        self.bwt_list = [float("-inf") for i in range(self.args.task_num)]
        self.avg_acc_list = [-1 for i in range(self.args.task_num)]
        self.whole_acc_list = [-1 for i in range(self.args.task_num)]
        self.temp_fwt = [float("-inf") for i in range(self.args.task_num)]
        self.fwt_list = [float("-inf") for i in range(self.args.task_num)]
        self.acc_rand_list = [0.0 for i in range(self.args.task_num)]

        self.model_save_path = model_save_path
        save_args(args, os.path.join(self.model_save_path, "config.json"))

        self.task_keys = []
        self.generation_arguments = {
            'max_new_tokens': None,
            'min_length': 5,
            'temperature': 1.0,
            'do_sample': False,
            'top_k': 0,
            'top_p': 0.9,
            'repetition_penalty': 1.0,
            'num_beams': 4,
            'bad_words_ids': [[628], [198]]
        }

        adapters.init(self.model)
        self.root_adapter_path = self.args.root_adapter_path
        self.adapter_path = self.root_adapter_path

        self.key_belong = None

    def _batchify(self, examples, batch_size):
        for i in range(0, len(examples), batch_size):
            yield examples[i:i + batch_size]
    
    def change_grad_true(self,task_id : int) -> None:
        for key_param in self.task_keys[task_id]:
            key_param.requires_grad = True

    def change_grad_false(self,task_id : int) -> None:
        for key_param in self.task_keys[task_id]:
            key_param.requires_grad = False

    def initialize_adapter_pool(self, task_id, adapter_skeleton) -> dict:
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
            elif self.args.key_initialize_method == "question+SQL":
                print(sample)
                text = sample["question_masked"] + "||" + sample["sql_masked"]
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
        
        if self.args.initialize_pool_method == "special_kmeans":
            #determine the key shoule be chosen by this task
            self.key_belong = []
                    # Determine the optimal number of clusters using silhouette score
            silhouette_scores = []
            for n_clusters in range(2, min(len(data_samples), 8)):  # Test a range of cluster numbers
                kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                labels = kmeans.fit_predict(all_embeddings_tensor.cpu().numpy())
                score = silhouette_score(all_embeddings_tensor.cpu().numpy(), labels)
                silhouette_scores.append(score)
            
            optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because range starts from 2
            print("task {} optimal_clusters : {}".format(task_id,optimal_clusters))
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
            kmeans.fit(all_embeddings_tensor.cpu().numpy())
            cluster_centers = kmeans.cluster_centers_

            # Initialize keys based on cluster centers
            for center in cluster_centers:
                center_tensor = torch.tensor(center, dtype=torch.float32, device=self.device)
                task_embedding_param = nn.Parameter(center_tensor.unsqueeze(0))
                self.task_keys.append(task_embedding_param)

                # Create adapter and directory
                adapter_name = adapter_skeleton + str(len(self.task_keys) - 1)
                self.key_belong.append(len(self.task_keys) - 1)
                self.adapter_path = os.path.join(self.root_adapter_path, adapter_name)
                if not os.path.exists(self.adapter_path):
                    os.makedirs(self.adapter_path, exist_ok=True)
                self.add_adapter(adapter_name)
                self.save_delete_adapter(adapter_name)
        
        else:
            embeddings_np = all_embeddings_tensor.cpu().numpy()
            kmeans = KMeans(n_clusters=key_num, random_state=0)
            clusters = kmeans.fit_predict(embeddings_np)
            cluster_centers = kmeans.cluster_centers_

            closest_data_indices = np.zeros(cluster_centers.shape[0], dtype=int)

            for center_idx, center in enumerate(cluster_centers):
                distances = np.linalg.norm(embeddings_np - center, axis=1)
                closest_data_indices[center_idx] = np.argmin(distances)

            center_now = list(closest_data_indices)
            store_sample = []
            for idx in center_now:
                store_sample.append(data_samples[idx])
            
            with open("store_sample.json",'a') as fp:
                json.dump(store_sample,fp)

            if self.args.initialize_pool_method == "random" or task_id == 0:
                print("ok")
                for center in cluster_centers:
                    center_tensor = torch.tensor(center, dtype=torch.float32, device=self.device)
                    task_embedding_param = nn.Parameter(center_tensor.unsqueeze(0))
                    self.task_keys.append(task_embedding_param)

                    # Create adapter and directory
                    adapter_name = adapter_skeleton + str(len(self.task_keys) - 1)
                    self.adapter_path = os.path.join(self.root_adapter_path, adapter_name)
                    if not os.path.exists(self.adapter_path):
                        os.mkdir(self.adapter_path)
                    self.add_adapter(adapter_name)
                    self.save_delete_adapter(adapter_name)
            elif self.args.initialize_pool_method == "inherit":
                existing_keys = torch.cat([key.data for key in self.task_keys],dim=0)
                for center in cluster_centers:
                    center_tensor = torch.tensor(center, dtype=torch.float32, device=self.device)
                    scores = self.get_scores(center_tensor.unsqueeze(0), existing_keys).squeeze(0)
                    k = min(5, scores.size(0))
                    _, top_indices = scores.topk(k)

                    # Compute the result of dividing indices by the pool size
                    div_results = top_indices // self.args.initialize_pool_size
                    max_result = div_results.max()
                    best_indices = top_indices[div_results == max_result]

                    best_key_index = best_indices[0].item()  # Select the first index if there are ties
                    best_key_name = adapter_skeleton + str(best_key_index)  # Example adapter name convention

                    # Load the best matched adapter
                    new_adapter_name = adapter_skeleton + str(len(self.task_keys))
                    self.load_adapter(best_key_name,load_as = new_adapter_name)
                    
                    self.task_keys.append(nn.Parameter(center_tensor.unsqueeze(0)))
                    self.adapter_path = os.path.join(self.root_adapter_path, new_adapter_name)
                    if not os.path.exists(self.adapter_path):
                        os.makedirs(self.adapter_path, exist_ok=True)
                    self.save_adapter(new_adapter_name)  # Save the new configuration under a new name
                    self.delete_adapter(new_adapter_name)  # Optional: remove the best_key adapter if not needed anymore
            else:
                raise NotImplementedError("Undefined initialize pool method: {}".format(self.args.initialize_pool_method))
    
    def prepare_data(self, example: dict):
        information = example
        text = example["text"]
        sql = information["sql"]
        return text,sql
    
    def make_optimizer_groups_forT5(self):
        # 
        if self.args.frozen_mode == "frozen_plm":
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if p.requires_grad], 'lr': self.args.lr}
            ]
        else:
            optimizer_grouped_parameters = [
                {'params': self.model.parameters(), 'lr': self.args.lr}
            ]
        return optimizer_grouped_parameters

    def make_optimizer_groups(self,task_id):
        # 
        if task_id not in self.task_keys:
            raise ValueError(f"Task ID {task_id} is not valid or not found in task_keys")
        if self.args.frozen_mode == "frozen_plm":
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if p.requires_grad], 'lr': self.args.lr}
            ]
            optimizer_grouped_parameters.append({'params': [key_param for key_param in self.task_keys[task_id]], 'lr': self.args.lr})
        else:
            optimizer_grouped_parameters = [
                {'params': self.model.parameters(), 'lr': self.args.lr}
            ]
            optimizer_grouped_parameters.append({'params': [key_param for key_param in self.task_keys[task_id]], 'lr': self.args.lr})
        return optimizer_grouped_parameters
    
    def save(self, model, name="model.bin"):
        torch.save({"model": model.state_dict()}, open(os.path.join(self.model_save_path, name), 'wb'))

    def load(self, model, name="model.bin", path=None):
        if path is None:
            model.load_state_dict(torch.load(open(os.path.join(self.model_save_path, name), "rb"), map_location=self.device)["model"])
        else:
            model.load_state_dict(torch.load(open(os.path.join(path, name), "rb"), map_location=self.device)["model"])


    def train_one_batch(self, examples, report_loss, example_num):
        input_texts = []
        target_texts = []
        for example in examples:
            text, sql = self.prepare_data(example)
            input_texts.append(text)
            target_texts.append(sql)

        inputs = self.tokenizer(input_texts, max_length=self.args.encoder_dim, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(target_texts, max_length=self.args.encoder_dim, padding=True, truncation=True, return_tensors='pt').to(self.device)

        decoder_input_ids = targets['input_ids']

        labels = targets['input_ids']
        labels[targets['attention_mask'] == 0] = -100

        outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels)

        loss = outputs.loss

        report_loss += loss.item() * inputs['input_ids'].size(0)
        example_num += inputs['input_ids'].size(0)

        return report_loss, example_num, loss

    def train_one_epoch(self, examples, optimize_step=True):
        self.model.train()

        random.shuffle(examples)
        report_loss = 0.0
        st = 0
        cnt = 0
        example_num = 0

        self.optimizer.zero_grad()
        while st < len(examples):
            ed = st + self.args.batch_size if st + self.args.batch_size < len(examples) else len(examples)

            report_loss, example_num, loss = self.train_one_batch(examples[st:ed], report_loss, example_num)
            loss.backward()

            if (cnt + 1) % self.args.accumulation_step == 0 or ed == len(examples):
                if self.args.clip_grad > 0.:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                if optimize_step:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            st = ed
            cnt += 1

        avg_loss = report_loss / example_num
        return avg_loss

    def add_adapter(self,adapter_name : str) -> None:
        self.model.add_adapter(adapter_name)
        self.model.set_active_adapters(adapter_name)
        self.model.to(self.device)
    
    def save_delete_adapter(self, adapter_name : str) -> None:
        self.model.save_adapter(self.adapter_path, adapter_name)
        self.model.delete_adapter(adapter_name)

    def save_adapter(self, adapter_name : str) -> None:
        self.model.save_adapter(self.adapter_path, adapter_name)
    
    def delete_adapter(self, adapter_name : str) -> None:
        self.model.delete_adapter(adapter_name)
    
    def load_adapter(self,adapter_name : str, load_as = None) -> None:
        self.model.load_adapter(os.path.join(self.root_adapter_path,adapter_name),load_as = load_as)
        print("loading from{}".format(os.path.join(self.root_adapter_path,adapter_name)))
        if load_as:
            self.model.set_active_adapters(load_as)
        else:
            self.model.set_active_adapters(adapter_name)
        self.model.to(self.device)

    #get the score of input query and key
    def get_scores(self, queries, keys, type='cosine'):
        if type == 'cosine':
            queries = torch.nn.functional.normalize(queries, p=2.0, dim=1)
            keys = torch.nn.functional.normalize(keys, p=2.0, dim=1)
            cos_sim = queries.mm(keys.T)
        else:
            raise NotImplementedError(f"Score type: {type} not implemented!!")

        assert (cos_sim>=-1).sum() == cos_sim.numel(), f'cosine similarity below -1'
        assert (cos_sim<=1).sum() == cos_sim.numel(), f'cosine similarity over 1'

        return cos_sim
    
    def get_queries(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        encoder_outputs = self.encoder_model.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        mask = inputs['attention_mask'].unsqueeze(-1).expand(encoder_outputs.last_hidden_state.size()).float()
        masked_outputs = encoder_outputs.last_hidden_state * mask
        sum_outputs = masked_outputs.sum(dim=1)
        sum_mask = mask.sum(dim=1)
        mean_outputs = sum_outputs / sum_mask
        return mean_outputs
    
    def make_train_classification(self,examples,task_id) -> dict:
        #first get all the queries
        if self.args.key_initialize_method == "question":
            texts_to_embedding = [example['text'].split('|')[1] for example in examples]
        elif self.args.key_initialize_method == "text":
            texts_to_embedding = [example['text'] for example in examples]
        elif self.args.key_initialize_method == "question+SQL":
            texts_to_embedding = [example["question_masked"] + "||" + example["sql_masked"] for example in examples]
        else:
            raise NotImplementedError("Undefined key initialize method: {}".format(self.args.key_initialize_method))

        examples_classified = {}
        batch_size = 16
        for key in self.task_keys:
            key.requires_grad = False
        for i in range(0, len(texts_to_embedding), batch_size):
            batch_texts = texts_to_embedding[i:i+batch_size]
            embeddings = self.get_queries(batch_texts)
            embeddings = torch.unbind(embeddings, dim=0)

            for exam_idx, embedding in enumerate(embeddings):
                original_idx = i + exam_idx
                query = embedding.unsqueeze(0)
                max_score = -1
                class_of_this = -1
                if self.key_belong:
                    for key_idx in self.key_belong:
                        key = self.task_keys[key_idx]
                        score = self.get_scores(query, key)
                        if score > max_score:
                            max_score = score
                            class_of_this = key_idx
                else:
                    for key_idx in range(task_id * self.args.initialize_pool_size, (task_id + 1) * self.args.initialize_pool_size):
                        key = self.task_keys[key_idx]
                        score = self.get_scores(query, key)
                        if score > max_score:
                            max_score = score
                            class_of_this = key_idx
                if class_of_this not in examples_classified:
                    examples_classified[class_of_this] = []
                examples_classified[class_of_this].append(examples[original_idx])

        return examples_classified
    
    def get_key_loss(self,examples,key,total_scores):
        if self.args.key_initialize_method == "question":
            texts = [example['text'].split('|')[1] for example in examples] 
        elif self.args.key_initialize_method == "text":
            texts = [example['text'] for example in examples] 
        elif self.args.key_initialize_method == "question+SQL":
            texts = [example["question_masked"] + "||" + example["sql_masked"] for example in examples]
        else:
            raise NotImplementedError("Undefined key initialize method: {}".format(self.args.key_initialize_method))
        
        batch_queries = self.get_queries(texts)
        scores = self.get_scores(batch_queries, key)
        prompt_loss = -1 * self.args.pool_lambda * scores.sum()
        score_sum = scores.detach().sum().item()
        total_scores += score_sum
        return  total_scores , prompt_loss
    

    def train_one_epoch_with_adapter(self, examples, adapter_name, key_idx, optimize_step=True):
        # Freeze T5 parameters
        for param in self.model.base_model.parameters():
            param.requires_grad = False
        
        # Ensure the newly added adapter is trainable
        self.model.train_adapter(adapter_name)

        self.task_keys[key_idx].requires_grad = True
        adapter_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if p.requires_grad], 'lr': self.args.adapter_lr}
            ]
        #what's the learning rate to set here?
        adapter_parameters.append({'params': [self.task_keys[key_idx]], 'lr': self.args.adapter_lr})
        adapter_optimizer = torch.optim.Adam(adapter_parameters, lr=self.args.adapter_lr)

        random.shuffle(examples)
        report_loss = 0.0
        total_scores = 0.0
        st = 0
        cnt = 0
        example_num = 0

        adapter_optimizer.zero_grad()
        while st < len(examples):
            ed = st + self.args.batch_size if st + self.args.batch_size < len(examples) else len(examples)

            report_loss, example_num, model_loss = self.train_one_batch(examples[st:ed], report_loss, example_num)

            total_scores,key_loss = self.get_key_loss(examples[st:ed],self.task_keys[key_idx],total_scores)
                       
            loss = key_loss + model_loss

            loss.backward()

            if (cnt + 1) % self.args.accumulation_step == 0 or ed == len(examples):
                if self.args.clip_grad > 0.:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                if optimize_step:
                    adapter_optimizer.step()
                    adapter_optimizer.zero_grad()

            st = ed
            cnt += 1

        avg_loss = report_loss / example_num
        avg_score = total_scores / example_num
        self.task_keys[key_idx].requires_grad = False
        return avg_loss, avg_score


    def batch_epoch_acc(self, examples, batch_size=32, save_prediction=False, task_now=0, test_on=0):
        self.model.eval()
        total_examples = len(examples)
        correct_predictions = 0

        right_result = []
        wrong_result = []
        all_result = []

        for batch_start in range(0, total_examples, batch_size):
            batch_end = min(batch_start + batch_size, total_examples)
            current_batch = examples[batch_start:batch_end]

            texts, sqls = zip(*[self.prepare_data(example) for example in current_batch])
            db_ids = [example['example']['db_id'] for example in current_batch]

            input_encodings = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(input_ids=input_encodings['input_ids'], attention_mask=input_encodings['attention_mask'], max_length=self.args.max_generate_length, **self.generation_arguments)
            
            predictions = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]

            for i, prediction in enumerate(predictions):
                all_result.append({
                    "db_id": db_ids[i],
                    "example": texts[i],
                    "sql": sqls[i],
                    "prediction": prediction
                })
                if "".join(prediction.strip().split()) == "".join(sqls[i].strip().split()):
                    correct_predictions += 1
                    right_result.append((current_batch[i], sqls[i], prediction))
                else:
                    wrong_result.append((current_batch[i], sqls[i], prediction))

        if save_prediction:
            with open(os.path.join(self.args.prediction_save_path, "task_" + str(task_now) + "_test_on_task_" + str(test_on) + ".json"), 'w') as f:
                json.dump(all_result, f)
        
        accuracy = correct_predictions / total_examples if total_examples > 0 else 0
        return accuracy, None, (right_result, wrong_result, None), None



    def epoch_acc(self, examples, name_skeleton, batch_size=32, cur_task=True, save_prediction = False,task_now = 0,test_on = 0):
        self.model.eval()
        correct_predictions = 0

        right_result = []
        wrong_result = []
        all_result = {}
        if_alright = True
        
        classification = [[] for _ in range(len(self.task_keys))]

        #to calculate the choose right rate
        chosen_right_rate = 0
        choose_all = len(examples)
        choose_right = 0
        
        for example_id,example in tqdm(enumerate(examples)):
            text, sql = self.prepare_data(example)
            information = example
            db_id_now = information['example']['db_id']

            query = self.get_queries(text)
            max_score = -1
            chose_key = -1
            for idx in range(len(self.task_keys)):
                self.task_keys[idx].requires_grad = False
                score = self.get_scores(query, self.task_keys[idx], type='cosine')
                if(score > max_score):
                    max_score = score
                    chose_key = idx
            classification[chose_key].append({
                "text" : text,
                "sql" : sql,
                "db_id" : db_id_now,
                "example_id" : example_id
            })
        print("epoch_acc classification situration:")
        for idx,item in enumerate(classification):
            print(len(item))
            if idx >= test_on * self.args.initialize_pool_size and idx < ( test_on + 1 ) * self.args.initialize_pool_size:
                choose_right += len(item)
        chosen_right_rate = choose_right / choose_all

        for key,child_class in enumerate(classification):
            child_examples = len(child_class)
            self.load_adapter(name_skeleton + str(key))
            for batch_start in range(0, child_examples, batch_size):
                batch_end = min(batch_start + batch_size, child_examples)
                current_batch = child_class[batch_start:batch_end]

                texts, sqls = zip(*[self.prepare_data(example) for example in current_batch])
                db_ids = [example['db_id'] for example in current_batch]
                example_ids = [example['example_id'] for example in current_batch]

                input_encodings = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(input_ids=input_encodings['input_ids'], attention_mask=input_encodings['attention_mask'], max_length=self.args.max_generate_length, **self.generation_arguments)
                
                predictions = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]

                for i, prediction in enumerate(predictions):
                    all_result[example_ids[i]] = {
                        "db_id": db_ids[i],
                        "example": texts[i],
                        "sql": sqls[i],
                        "prediction": prediction
                    }
                    if "".join(prediction.strip().split()) == "".join(sqls[i].strip().split()):
                        correct_predictions += 1
                        right_result.append((current_batch[i], sqls[i], prediction))
                    else:
                        wrong_result.append((current_batch[i], sqls[i], prediction))

            self.delete_adapter(name_skeleton + str(key))
        results_output = []
        for i in range(len(examples)):
            results_output.append(all_result[i])
        
        if save_prediction:
            with open(os.path.join(self.args.prediction_save_path,"task_" + str(task_now) + "_test_on_task_" + str(test_on) + ".json"), 'w') as f:
                json.dump(results_output,f)
        accuracy = correct_predictions / len(examples) if len(examples) > 0 else 0
        return accuracy, chosen_right_rate, (right_result, wrong_result, None), None

    def eval_task_stream(self, task_id, name_skeleton, plan_choice="B"):
        if plan_choice == "B":
            choose_rate_list = []
            for k in range(task_id + 1):
                test_acc, chosen_right_rate, (right, wrong, _), _ = self.epoch_acc(self.task_controller.task_list[k]["test"],name_skeleton,save_prediction = True, task_now = task_id,test_on = k)
                choose_rate_list.append(chosen_right_rate)
                print("--- Task", k, "Test Accuracy:", test_acc)
            if task_id < self.args.task_num - 1:
                test_acc, chosen_right_rate, (right, wrong, _), _ = self.epoch_acc(self.task_controller.task_list[task_id + 1]["test"],name_skeleton,save_prediction = True, task_now = task_id,test_on = task_id + 1)
                choose_rate_list.append(chosen_right_rate)
                print("--- Task", task_id + 1, "Test Accuracy:", test_acc)
            return choose_rate_list
        elif plan_choice == "A":
            if task_id == 0:
                for k in range(task_id + 2):
                    test_acc, _, (right, wrong, _), _ = self.batch_epoch_acc(self.task_controller.task_list[k]["test"], batch_size=16,save_prediction = True, task_now = task_id,test_on = k)
                    print("--- Task", k, "Test Accuracy:", test_acc)
            else:
                for k in range(task_id + 1):
                    if k==0:
                        test_acc, _, (right, wrong, _), _ = self.batch_epoch_acc(self.task_controller.task_list[k]["test"], batch_size=16,save_prediction = True, task_now = task_id,test_on = k)
                    else:
                        self.load_adapter(name_skeleton + str(k))
                        test_acc, _, (right, wrong, _), _ = self.batch_epoch_acc(self.task_controller.task_list[k]["test"], batch_size=16,save_prediction = True, task_now = task_id,test_on = k)
                        self.delete_adapter(name_skeleton + str(k))
                    print("--- Task", k, "Test Accuracy:", test_acc)
                if task_id < self.args.task_num - 1:
                    self.load_adapter(name_skeleton + str(task_id))
                    test_acc, _, (right, wrong, _), _ = self.batch_epoch_acc(self.task_controller.task_list[task_id + 1]["test"], batch_size=16,save_prediction = True, task_now = task_id,test_on = task_id + 1)
                    self.delete_adapter(name_skeleton + str(task_id))
                    print("--- Task", task_id + 1, "Test Accuracy:", test_acc)
