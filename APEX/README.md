## APEX
APEX model training and testing code for our paper: [Discarding the Crutches: Adaptive Parameter-Efficient Expert Meta-Learning for Continual Semantic Parsing]

### Model
Put **t5-base-lm-adapt** into the 'model' directory
You can download the model [here](https://huggingface.co/google/t5-base-lm-adapt)
if you want to use other models, put them into the 'model' directory too. In our paper, we also use t5-large and t5-3b.
You can download them [here](https://huggingface.co/google-t5/t5-large) and [here](https://huggingface.co/google-t5/t5-3b)

### Data
Put augmented **combine_task_stream** and **spider_task_stream** into the 'data' directory
These datasets were proposed by this [paper](https://arxiv.org/abs/2310.04801)
You can download them and augment them using our method proposed by our paper.

### Running Code

#### Train
We provide you with some scripts that you can execute directly to train the model. For example, if you want to use t5-base to train with adapters on the combine order0 dataset, you can execute this:
```bash
cd APEX
sh train_scripts/with_cl/base_combine_adapter_order0.sh
```
You may need to modify some parameters:
```
--task_path             #where to load the dataset
--root_adapter_path     #where to save the adapters
--prediction_save_path  @where to save the result
```

#### Predict
For example, if you want to predict the model trained using t5-base with adapters on the combine order0 dataset, you can execute this:
```bash
sh train_scripts/with_cl/load_test/base_combine_adapter_order0.sh
```
You may need to modify some parameters:
```
--task_path             #where to load the dataset
--model_load_path       #where to load the trained model
--root_adapter_path     #where to save the adapters
--prediction_save_path  #where to save the result
```
These parameters should be the same as the model were trained

#### Evaluate
Evaluating on the combine stream dataset
```bash
python evaluate/on_combine/evaluate_combine.py
```

Evaluating on the combine stream dataset
```bash
python data_out_process.py
python files_to_em_and_ex.py
```

### Relevant Repositories
This codebase has some code or ideas ported from the following repositories.
1. [SSCL-Text2SQL](https://github.com/Bahuia/SSCL-Text2SQL)
2. [C3](https://github.com/Bahuia/C3)
3. [spider](https://github.com/taoyds/spider)