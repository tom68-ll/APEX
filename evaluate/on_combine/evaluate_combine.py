from train_utils.spider.evaluation import evaluate as spider_evaluate
from train_utils.wikisql.evaluation import evaluate as wikisql_evaluate
import json
import os
import numpy as np

tgt_path = "./result/20240929/T53B/T53B_sft"
order = "order0"
task_num = 7


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

wikisql_idx = [0,5,6]   #1,3,5
spider_idx = [1,2,3,4]  #0,2,4,6


if __name__ == '__main__':
    em_acc = np.zeros((task_num,task_num))
    ex_acc = np.zeros((task_num,task_num))
    task_test_num = {}
    if order == "order0":
        for i in range(task_num):
            for j in range(i+1):
                load_path = os.path.join(tgt_path,f"task_{i}_test_on_task_{j}.json")
                with open(load_path, 'r') as fp:
                    datas = json.load(fp)
                gold_sql = []
                gold_db = []
                predict = []
                for item in datas:
                    gold_sql.append(item["sql"])
                    gold_db.append(item["db_id"])
                    predict.append(item["prediction"])
                if j % 2 == 0:
                    score_exec, score_em, results = spider_evaluate(gold_sql, gold_db, predict, "all")
                elif j % 2 == 1:
                    score_exec, score_em, results = wikisql_evaluate(gold_sql, gold_db, predict, "all")
                em_acc[i][j] = score_em
                ex_acc[i][j] = score_exec
                print("now process ing {} on {}".format(i,j))
            task_test_num[i] = len(datas)
            if i < task_num - 1:
                j = i + 1
                load_path = os.path.join(tgt_path,f"task_{i}_test_on_task_{j}.json")
                with open(load_path, 'r') as fp:
                    datas = json.load(fp)
                gold_sql = []
                gold_db = []
                predict = []
                for item in datas:
                    gold_sql.append(item["sql"])
                    gold_db.append(item["db_id"])
                    predict.append(item["prediction"])
                if j % 2 == 0:

                    score_exec, score_em, results = spider_evaluate(gold_sql, gold_db, predict, "all")
                elif j % 2 == 1:
                    score_exec, score_em, results = wikisql_evaluate(gold_sql, gold_db, predict, "all")
                em_acc[i][j] = score_em
                ex_acc[i][j] = score_exec
                print("now process ing {} on {}".format(i,j))
    elif order == "order1":

        for i in range(task_num):
            for j in range(i+1):
                load_path = os.path.join(tgt_path,f"task_{i}_test_on_task_{j}.json")
                with open(load_path, 'r') as fp:
                    datas = json.load(fp)
                gold_sql = []
                gold_db = []
                predict = []
                for item in datas:
                    gold_sql.append(item["sql"])
                    gold_db.append(item["db_id"])
                    predict.append(item["prediction"])
                #if j % 2 == 0:
                if j in spider_idx:
                    score_exec, score_em, results = spider_evaluate(gold_sql, gold_db, predict, "all")
                #elif j % 2 == 1:
                else:
                    score_exec, score_em, results = wikisql_evaluate(gold_sql, gold_db, predict, "all")
                em_acc[i][j] = score_em
                ex_acc[i][j] = score_exec
                print("now process ing {} on {}".format(i,j))
            task_test_num[i] = len(datas)
            if i < task_num - 1:
                j = i + 1
                load_path = os.path.join(tgt_path,f"task_{i}_test_on_task_{j}.json")
                with open(load_path, 'r') as fp:
                    datas = json.load(fp)
                gold_sql = []
                gold_db = []
                predict = []
                for item in datas:
                    gold_sql.append(item["sql"])
                    gold_db.append(item["db_id"])
                    predict.append(item["prediction"])
                #if j % 2 == 0:
                if j in spider_idx:
                    score_exec, score_em, results = spider_evaluate(gold_sql, gold_db, predict, "all")
                #elif j % 2 == 1:
                else:
                    score_exec, score_em, results = wikisql_evaluate(gold_sql, gold_db, predict, "all")
                em_acc[i][j] = score_em
                ex_acc[i][j] = score_exec
                print("now process ing {} on {}".format(i,j))
    print('em_acc:')
    print(em_acc)
    print('ex_acc:')
    print(ex_acc)
    print('count:')
    print(task_test_num)
    stream_task_result = eval_stream_task(task_num,task_test_num, em_acc, ex_acc)
    for key in stream_task_result.keys():
        print(key+' :   ',stream_task_result[key])
    with open(os.path.join(tgt_path,"output.txt"), "w") as file:
        file.write('em_acc:\n')
        file.write(str(em_acc) + '\n')
        file.write('ex_acc:\n')
        file.write(str(ex_acc) + '\n')
        file.write('count:\n')
        file.write(str(task_test_num) + '\n')

        stream_task_result = eval_stream_task(task_num, task_test_num, em_acc, ex_acc)
        for key in stream_task_result.keys():
            file.write(f"{key} :   {stream_task_result[key]}\n")
