import argparse
import torch
import random
import numpy as np


def init_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--seed', default=5783287, type=int, help='random seed')
    arg_parser.add_argument('--cuda', action='store_true', help='use gpu')

    arg_parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    arg_parser.add_argument('--accumulation_step', default=4, type=int, help='Gradient Accumulation')

    arg_parser.add_argument('--beam_size', default=1, type=int, help='beam size for beam search')
    arg_parser.add_argument('--column_pointer', action='store_true', help='use column pointer')

    arg_parser.add_argument('--plm_model', default='bert-base-uncased', type=str, help='plm_model')
    arg_parser.add_argument('--encode_model', default='/home/jyzhang/user1/Huggingface_model/t5-base-lm-adapt', type=str, help='encode_model')
    arg_parser.add_argument('--encoder_dim', default=768, type=int, help='size of encoder_dim')

    arg_parser.add_argument('--action_embed_size', default=64, type=int, help='size of word embeddings')
    arg_parser.add_argument('--type_embed_size', default=32, type=int, help='size of word embeddings')
    arg_parser.add_argument('--hidden_size', default=300, type=int, help='size of LSTM hidden states')
    arg_parser.add_argument('--att_vec_size', default=300, type=int, help='size of attentional vector')
    arg_parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate')

    # readout layer
    arg_parser.add_argument('--readout', default='identity', choices=['identity', 'non_linear'])
    arg_parser.add_argument('--column_att', choices=['dot_prod', 'affine'], default='affine')

    arg_parser.add_argument('--decode_max_time_step', default=40, type=int, help='maximum number of time steps used '
                                                                                 'in decoding and sampling')

    arg_parser.add_argument('--clip_grad', default=5., type=float, help='clip gradients')
    arg_parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
    arg_parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    arg_parser.add_argument('--lr_prompt', default=0.001, type=float, help='learning rate of prompt')
    arg_parser.add_argument('--lr_plm', default=2e-5, type=float, help='learning rate of PLM')

    arg_parser.add_argument('--epoch', default=50, type=int, help='Maximum Epoch')

    arg_parser.add_argument('--task_num', type=int, default=10)
    arg_parser.add_argument('--task_path', type=str, default="../task_splits/spider_task_stream/task_{}/{}")
    arg_parser.add_argument('--combine_K', type=int, default=1)
    arg_parser.add_argument('--memory_size', type=int, default=20)
    arg_parser.add_argument('--candidate_size', type=int, default=50)
    arg_parser.add_argument('--device', type=str, default="0")

    arg_parser.add_argument('--baseline_name', type=str, default="ewc")
    arg_parser.add_argument('--vocab_path', type=str, default="./vocab.pkl")
    arg_parser.add_argument('--ewc_reg', type=float, default=1.)
    arg_parser.add_argument('--emar_second_iter', type=int, default=1)
    arg_parser.add_argument('--gem_margin', type=float, default=1.)
    arg_parser.add_argument('--setnet_it', type=float, default=2)
    arg_parser.add_argument('--mt_decay', type=float, default=0.1)
    arg_parser.add_argument('--n_way', type=int, default=1)
    arg_parser.add_argument('--k_shot', type=int, default=4)
    arg_parser.add_argument('--meta_task_num', type=int, default=100)

    arg_parser.add_argument('--soft_prompt_len', type=int, default=0)
    arg_parser.add_argument('--frozen_mode', type=str, choices=['frozen_plm', 'tuning_plm'])
    arg_parser.add_argument('--tuning_decoder', action='store_true')
    arg_parser.add_argument('--save_decoder', action='store_true')
    arg_parser.add_argument('--use_adaptation', action='store_true')
    arg_parser.add_argument('--adaptation_cpt', type=str, default="")
    arg_parser.add_argument('--use_adafactor', action='store_true')
    arg_parser.add_argument('--T5_patience', type=int, default=10)
    arg_parser.add_argument('--adapter_patience', type=int, default=20)
    arg_parser.add_argument('--T5_eval_epoch', type=int, default=50)
    arg_parser.add_argument('--adapter_eval_epoch', type=int, default=100)

    arg_parser.add_argument('--start_cpt', type=str, default="")

    arg_parser.add_argument('--use_demo', action='store_true')
    arg_parser.add_argument('--hard_prompt_threshold', type=float, default=0.7)
    arg_parser.add_argument('--hard_prompt_max_num', type=int, default=2)

    arg_parser.add_argument('--few_shot', type=int, default=50)
    arg_parser.add_argument('--max_seq_length', type=int, default=1024)

    arg_parser.add_argument('--task_perm', type=str, default='')

    arg_parser.add_argument('--max_source_length', type=int, default=1024)
    arg_parser.add_argument('--max_target_length', type=int, default=256)
    arg_parser.add_argument('--max_generate_length', type=int, default=256)

    arg_parser.add_argument('--prediction_save_path', type=str, default='/home/jyzhang/user1/nips-2023-supplementary_material/code/continual-text2sql-baselines/Myoutput')
    arg_parser.add_argument('--root_adapter_path', type=str)

    arg_parser.add_argument('--adapter_epoch',type=int, default = 100)
    arg_parser.add_argument('--adapter_lr',type=float,default=0.3)

    arg_parser.add_argument('--prompt_length',type=int,default=300)
    arg_parser.add_argument('--prefix_length',type=int,default=50)

    arg_parser.add_argument('--pool_lambda',type=float,default=0.1)
    arg_parser.add_argument('--initialize_pool_size',type=int,default=5)
    arg_parser.add_argument('--initialize_pool_method',type=str,default="random")
    arg_parser.add_argument('--key_initialize_method', type=str, default="question")
    arg_parser.add_argument('--model_load_path',type=str,default="")
    arg_parser.add_argument('--adapter_root_path',type=str,default="")

    arg_parser.add_argument('--load_test_trigger',type=str,default="",choices=["all_load","all_load_mix","choose_load_mix","choose_load_mix_two","have_a_try","ideal_all_load","get_zscores","test_T5", "ideal_all_load_ver2"])
    arg_parser.add_argument('--top_n',type=int,default=3)
    arg_parser.add_argument('--seperate_number',type=float,default=1.3)

    arg_parser.add_argument('--pet_type',type=str,default='adapter')        
    arg_parser.add_argument('--continue_task_id',type=int,default=0)        
    arg_parser.add_argument('--continue_time',type=str,default=None)        
    arg_parser.add_argument('--project_layer_path',type=str,default=None)
    arg_parser.add_argument('--first_augmentation', action='store_true', help='use augmentation only when training T5')

    args = arg_parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    return args