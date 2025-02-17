import json
import os
import argparse
import re




def remove_extra_spaces(input_string):
    # 使用正则表达式将连续的空格替换为单个空格
    processed_string = re.sub(r'\s+', ' ', input_string)
    return processed_string.strip()  # 去除首尾空格


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_name', type=str, help='whats ur model name')
    arg_parser.add_argument('--in_dir', type=str, default="/home/jyzhang/user1/nips-2023-supplementary_material/code/continual-text2sql-baselines/C3out/20241106_sfnet/llm_semi/mixtral")
    args = arg_parser.parse_args()
    in_dir = args.in_dir
    out_dir = os.path.join(in_dir,"process")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(os.path.join(out_dir,"source")):
        os.mkdir(os.path.join(out_dir,"source"))
    if not os.path.exists(os.path.join(out_dir,"target")):
        os.mkdir(os.path.join(out_dir,"target"))

    for task_id in range(10):
        if task_id != 9:
            range_to_search = task_id + 2
        else:
            range_to_search = task_id + 1
        for test_on in range(range_to_search):
            data_dir = os.path.join(in_dir,"task_" + str(task_id) + "_test_on_task_" + str(test_on) + ".json")
            with open(data_dir,'r') as f:
                datas = json.load(f)

            #加载数据
            pred = []
            gold = []
            for item in datas:
                pred.append(remove_extra_spaces(item["prediction"]))
                gold.append(remove_extra_spaces(item["sql"] + '||' + item["db_id"]))
            
            source_dir = os.path.join(out_dir,"source","task_" + str(task_id) + "_test_on_task_" + str(test_on) + ".src")
            target_dir = os.path.join(out_dir,"target","task_" + str(task_id) + "_test_on_task_" + str(test_on) + ".tgt")

            with open(source_dir,'w') as f:
                for item in pred:
                    f.writelines(item + '\n')
            with open(target_dir,'w') as f:
                for item in gold:
                    f.writelines(item + '\n')
            