import subprocess
import argparse
import os
import sys
import numpy as np
import json
import sqlite3
import spider.evaluation 
from spider.evaluation import evaluation_mine

def eval_stream_task(task_num: int, test_num: dict, em_acc, ex_acc,etype = 'all'):

    acc_avg_em = [float("-inf") for i in range(task_num)]
    acc_whole_em = [float("-inf") for i in range(task_num)]
    bwt_em = [float("-inf") for i in range(task_num)]
    fwt_em = [float("-inf") for i in range(task_num)]

    acc_avg_ex = [float("-inf") for i in range(task_num)]
    acc_whole_ex = [float("-inf") for i in range(task_num)]
    bwt_ex = [float("-inf") for i in range(task_num)]
    fwt_ex = [float("-inf") for i in range(task_num)]

    for task_id in range(task_num):
        # acc_avg AND acc_whole
        acc_avg_em[task_id] =  np.mean(em_acc[task_id, 0:task_id+1])
        acc_avg_ex[task_id] =  np.mean(ex_acc[task_id, 0:task_id+1])

        acc_whole_em[task_id] = np.sum(em_acc[task_id, 0:task_id+1] * np.array(list(test_num.values())[0:task_id+1])) / np.sum(np.array(list(test_num.values())[0:task_id+1]))
        acc_whole_ex[task_id] = np.sum(ex_acc[task_id, 0:task_id+1] * np.array(list(test_num.values())[0:task_id+1])) / np.sum(np.array(list(test_num.values())[0:task_id+1]))

        # BWT AND FWT
        if task_id > 0:
            # BWT
            bwt_em_tmp = 0
            for past_id in range(task_id):
                bwt_em_tmp += em_acc[task_id][past_id] - em_acc[past_id][past_id]
            bwt_em[task_id] = bwt_em_tmp / task_id

            bwt_ex[task_id] = 0
            for past_id in range(task_id):
                bwt_ex[task_id] += ex_acc[task_id][past_id] - ex_acc[past_id][past_id]
            bwt_ex[task_id] = bwt_ex[task_id] / task_id

            # FWT
            fwt_em_tmp = 0
            for i in range(task_id, -1, -1):
                fwt_em_tmp += em_acc[i-1][i]
                if i-1 == 0:
                    break
            fwt_em[task_id] = fwt_em_tmp / task_id

            fwt_ex_tmp = 0
            for i in range(task_id, -1, -1):
                fwt_ex_tmp += ex_acc[i-1][i]
                if i-1 == 0:
                    break
            fwt_ex[task_id] = fwt_ex_tmp / task_id

    stream_task_result = {
        "acc_avg_em": acc_avg_em, 
        "acc_avg_ex": acc_avg_ex, 
        "acc_whole_em": acc_whole_em, 
        "acc_whole_ex": acc_whole_ex,
        "bwf_em": bwt_em, 
        "bwf_ex": bwt_ex,
        "fwt_em": fwt_em, 
        "fwt_ex": fwt_ex
        }

    return stream_task_result


def make_all_spider_test(task_num,source_path,target_path,spider_path,etype='all'):
    etype == 'all'
    em_acc = np.zeros((task_num,task_num))
    ex_acc = np.zeros((task_num,task_num))
    task_test_num = {}
    for i in range(task_num):
        for j in range(i+1):
            gold_path = os.path.join(target_path,"task_" + str(i) + "_test_on_task_" + str(j) + ".tgt")
            pred_path = os.path.join(source_path,"task_" + str(i) + "_test_on_task_" + str(j) + ".src")
            db_path = os.path.join(spider_path,'database')
            table_path = os.path.join(spider_path,'tables.json')
            test_dict = evaluation_mine(gold=gold_path,pred=pred_path,db_dir=db_path,table=table_path,etype=etype)
            em_acc[i][j] = test_dict['match']
            ex_acc[i][j] = test_dict['exec']
            print("now process ing {} on {}".format(i,j))
        task_test_num[i] = test_dict['count']
        if i < task_num - 1:
            gold_path = os.path.join(target_path,"task_" + str(i) + "_test_on_task_" + str(i+1) + ".tgt")
            pred_path = os.path.join(source_path,"task_" + str(i) + "_test_on_task_" + str(i+1) + ".src")
            db_path = os.path.join(spider_path,'database')
            table_path = os.path.join(spider_path,'tables.json')
            test_dict = evaluation_mine(gold=gold_path,pred=pred_path,db_dir=db_path,table=table_path,etype=etype)
            em_acc[i][i+1] = test_dict['match']
            ex_acc[i][i+1] = test_dict['exec']
            print("now process ing {} on {}".format(i,i+1))
    print('em_acc:')
    print(em_acc)
    print('ex_acc:')
    print(ex_acc)
    print('count:')
    print(task_test_num)
    stream_task_result = eval_stream_task(task_num,task_test_num, em_acc, ex_acc)
    for key in stream_task_result.keys():
        print(key+' :   ',stream_task_result[key])

    
#gold,pred,db_dir,table,etype
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--source_path', type=str, help='task?teston? path', default='')
    arg_parser.add_argument('--target_path', type=str, help='original data path', default='')
    arg_parser.add_argument('--spider_path', type=str, help='spider\'s tables location',default='')
    arg_parser.add_argument('--task_num', type=int, help='task num',default=10)
    arg_parser.add_argument('--etype', default='all')
    args = arg_parser.parse_args()

    print("make_all_spider_test:\n")
    make_all_spider_test(args.task_num,args.source_path,args.target_path,args.spider_path,etype=args.etype)

