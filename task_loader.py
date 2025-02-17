# -*- coding: utf-8 -*-
# !/usr/bin/python

import sys
import random
import os
import copy
import json
import numpy as np
from nltk.stem import WordNetLemmatizer
sys.path.append("..")
from utils.dataset import Example
from rule.define_rule import Sup, Sel, Order, Root, Filter, A, N, C, T, Root1


AGG_OPS = ['none', 'max', 'min', 'count', 'sum', 'avg']
wordnet_lemmatizer = WordNetLemmatizer()


class TaskLoader(object):

    def __init__(self, args):
        self.args = args
        self.task_num = args.task_num
        self.memory_size = self.args.memory_size
        self.task_list = self.load_continual_tasks(args.task_path, args.task_perm, args.combine_K, args.few_shot)
        self.task_num = len(self.task_list)
        self.memory_list = []

        self.whole_test = [[] for _ in range(self.task_num)]
        for task in self.task_list:
            self.whole_test.extend(task["test"])

    def load_one_task(self, sql_path):
        sql_data = []

        print("Loading task_splits from %s" % sql_path)
        with open(sql_path) as inf:
            data = lower_keys(json.load(inf))
            sql_data += data

        return sql_data

    def load_continual_tasks(self, task_path, task_perm=None, combine_K=1, few_shot=50):

        if any(dataset in task_path for dataset in ['spider', 'cosql', 'sparc']):
            '''
            task_0 = [task_perm[:combine_K]]
            other_tasks = [[x] for x in task_perm[combine_K:]]
            random.shuffle(other_tasks)
            '''
            task_0 = [task_perm[:1]]
            other_tasks = [[x] for x in task_perm[1:]]
        elif any(dataset in task_path for dataset in ['combine1', 'combine2']):
            task_0 = [task_perm[:1]]
            other_tasks = [[x] for x in task_perm[1:]]
            '''
            task_0 = [task_perm[:combine_K]]
            other_tasks_1 = [[x] for i, x in enumerate(task_perm[combine_K:]) if i % 2 == 0]
            other_tasks_2 = [[x] for i, x in enumerate(task_perm[combine_K:]) if i % 2 == 1]
            random.shuffle(other_tasks_1)
            random.shuffle(other_tasks_2)
            other_tasks = [other_tasks_1[i // 2] if i % 2 == 0 else other_tasks_2[i // 2] for i in
                           range(len(task_perm[combine_K:]))]
            '''
        else:
            raise NotImplementedError("No such dataset !")

        task_split = task_0 + other_tasks
        print("Loading from datasets ...")
        print("Shot Number: ", few_shot)
        print("Task Order: ", task_split)

        raw_datasets = []
        whole_test_data = []

        def get_raw_dataset(task_ids):
            train_data = []
            dev_data = []
            test_data = []

            for task_id in task_ids:
                train_path = os.path.join(task_path.format(task_id, "train.json"))
                dev_path = os.path.join(task_path.format(task_id, "dev.json"))
                test_path = os.path.join(task_path.format(task_id, "test.json"))

                train_data += self.load_one_task(train_path)
                dev_data += self.load_one_task(dev_path)
                test_data += self.load_one_task(test_path)

            random.shuffle(train_data)
            random.shuffle(dev_data)
            random.shuffle(test_data)

            if self.args.use_demo:
                id2examples = {}
                for example in train_data:
                    id2examples[example.qid] = example

                add_demo(train_data, id2examples)
                add_demo(dev_data, id2examples)
                add_demo(test_data, id2examples)

            return train_data, dev_data, test_data

        def add_demo(data, id2examples):

            for example in data:
                cols_demo, tables_demo = get_prompt_examples(example, id2examples)
                assert cols_demo is not None
                assert tables_demo is not None
                example.hard_col_demo = cols_demo
                example.hard_table_demo = tables_demo

        for i, task_ids in enumerate(task_split):
            train_data, dev_data, test_data = get_raw_dataset(task_ids)

            if few_shot > 0:
                train_data = [x for x in train_data[:few_shot * len(task_ids)]]

            whole_test_data += [x for x in test_data]

            raw_dataset = {
                'train': train_data,
                'dev': dev_data,
                'test': test_data,
                'whole_test': [x for x in whole_test_data]
            }
            raw_datasets.append(raw_dataset)

        return raw_datasets

def get_prompt_examples(example, id2examples, topk=1):
    cols_demo = []
    table_demo = []
    cnt = 0
    for idx in example.demo_ex_ids[:topk]:
        if idx not in id2examples:
            continue
        cnt += 1
        demo = id2examples[idx]

        one_cols_demo = [" ".join(demo.src_sent), [" ".join(x) for x in demo.tab_cols]]
        one_tables_demo = [" ".join(demo.src_sent), [" ".join(x) for x in demo.table_names]]
        # print(demo.tab_cols)
        # print(demo.table_names)
        # print(one_cols_demo)
        # print(one_tables_demo)
        # exit()

        cols_demo.append(one_cols_demo)
        table_demo.append(one_tables_demo)

        if cnt >= topk:
            break

        # cols_demo += [" ".join(example_2.src_sent), [" ".join(demo.tab_cols[x]) for x in cols_2]]
        # tables_prompt = [[" ".join(example_2.src_sent), [" ".join(example_2.table_names[x]) for x in tables_2]] for example_2, tables_2 in new_tmp_tables]

    # print([" ".join(example.tab_cols[x]) for x in cols_1])
    # print(cols_prompt[:5])
    # print([" ".join(example.table_names[x]) for x in tables_1])
    # print(tables_prompt[:5])
    # exit()
    return cols_demo, table_demo


def lower_keys(x):
    if isinstance(x, list):
        return [lower_keys(v) for v in x]
    elif isinstance(x, dict):
        return dict((k.lower(), lower_keys(v)) for k, v in x.items())
    else:
        return x

def get_table_colNames(tab_ids, tab_cols):
    table_col_dict = {}
    for ci, cv in zip(tab_ids, tab_cols):
        if ci != -1:
            table_col_dict[ci] = table_col_dict.get(ci, []) + cv
    result = []
    for ci in range(len(table_col_dict)):
        result.append(table_col_dict[ci])
    return result