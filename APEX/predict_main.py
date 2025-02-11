# -*- coding: utf-8 -*-
# !/usr/bin/python

import os
import csv
import sys
sys.path.append("..")
from load_test_best_new.amtm import AMTMTrainer
from load_test_best_new.adapter_trainer import BasicTrainer
from load_test_for_T53B.amtm import AMTMTrainer_T53B
from args import init_arg_parser
import datetime

def init_log_checkpoint_path():
    dir_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(os.path.curdir, "saved_model", dir_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path, dir_name

def generate_result_file(out_path, avg_acc_list, whole_acc_list, bwt_list, fwt_list):
    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([""] + [i for i in range(len(avg_acc_list))])
        writer.writerow(["avg_acc"] + avg_acc_list)
        writer.writerow(["whole_acc"] + whole_acc_list)
        writer.writerow(["bwt"] + bwt_list)
        writer.writerow(["fwt"] + fwt_list)

def run_training():

    args = init_arg_parser()
    # print("Random Seed:", args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    print("task_num: {}".format(args.task_perm))

    if not os.path.exists(args.prediction_save_path):
        os.mkdir(args.prediction_save_path)

    if any(dataset in args.task_path for dataset in ['spider', 'cosql', 'sparc']):
        #args.task_perm = [11, 5, 3, 7, 12, 2, 10, 8, 6, 4, 15, 0, 1, 13, 14, 9]
        args.task_perm = [0,1,2,3,4,5,6,7,8,9,10]
    elif any(dataset in args.task_path for dataset in ['combine1']):
        args.task_perm = [0,1,2,3,4,5,6]
    elif any(dataset in args.task_path for dataset in ['combine2']):
        args.task_perm = [0, 1, 2, 3, 4, 5, 11, 6, 12, 7, 13, 8, 14, 9, 15, 10]
    else:
        raise NotImplementedError("No such dataset !")

    for k, v in sorted(vars(args).items()):
        if k == "task_perm":
            continue
        print(k, '=', v)

    model_save_path, dir_name = init_log_checkpoint_path()
    print("Current Training Data Will Be Saved in Path: {}".format(model_save_path))
    
    if args.baseline_name == "amtm":
        trainer = AMTMTrainer(args, model_save_path)
    elif args.baseline_name == "amtm_T53B":
        trainer = AMTMTrainer_T53B(args, model_save_path)
    elif args.baseline_name == "amtm_codeS":
        pass
        #trainer = AMTMTrainer_codeS(args, model_save_path)
    else:
        raise NotImplementedError

    avg_acc_list, whole_acc_list, bwt_list, fwt_list = trainer.train()
    print("Finish Continual Learning.")

    result_path = "./results/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    print("avg_acc_list:", avg_acc_list)
    print("whole_acc_list:", whole_acc_list)
    print("bwt_list:", bwt_list)
    print("fwt_list:", fwt_list)
    print()

if __name__ == "__main__":
    run_training()