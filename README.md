## APEX
The code for our COLING 2025 paper: **Discarding the Crutches: Adaptive Parameter-Efficient Expert Meta-Learning for Continual Semantic Parsing**.

### Model
Put **t5-base-lm-adapt** into the 'model' directory.
You can download the model [here](https://huggingface.co/google/t5-base-lm-adapt)
If you want to use other models, put them into the 'model' directory too. In our paper, we also use t5-large and t5-3b.
You can download them [here](https://huggingface.co/google-t5/t5-large) and [here](https://huggingface.co/google-t5/t5-3b)

### Data
Put augmented **combine_task_stream** and **spider_task_stream** into the 'data' directory.
These datasets were proposed by this [paper](https://arxiv.org/abs/2310.04801)
You can download them and augment them using our method proposed by our paper.

### Running Code

#### Train
We provide you with some scripts that you can execute directly to train the model. For example, if you want to use t5-base to train with adapters on the combined order0 dataset, you can execute this:
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
For example, if you want to predict the model trained using t5-base with adapters on the combined order0 dataset, you can execute this:
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
These parameters should be the same as the model was trained.

#### Evaluate
Evaluating on the combined stream dataset.
```bash
python evaluate/on_combine/evaluate_combine.py
```

Evaluating on the combined stream dataset.
```bash
python data_out_process.py
python files_to_em_and_ex.py
```

### Relevant Repositories
This codebase has some code or ideas ported from the following repositories.
1. [SSCL-Text2SQL](https://github.com/Bahuia/SSCL-Text2SQL)
2. [C3](https://github.com/Bahuia/C3)
3. [Spider](https://github.com/taoyds/spider)

### Citation
```
@inproceedings{liu-etal-2025-discarding,
    title = "Discarding the Crutches: Adaptive Parameter-Efficient Expert Meta-Learning for Continual Semantic Parsing",
    author = "Liu, Ruiheng  and
      Zhang, Jinyu  and
      Song, Yanqi  and
      Zhang, Yu  and
      Yang, Bailong",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.240/",
    pages = "3560--3578",
    abstract = "Continual Semantic Parsing (CSP) enables parsers to generate SQL from natural language questions in task streams, using minimal annotated data to handle dynamically evolving databases in real-world scenarios. Previous works often rely on replaying historical data, which poses privacy concerns. Recently, replay-free continual learning methods based on Parameter-Efficient Tuning (PET) have gained widespread attention. However, they often rely on ideal settings and initial task data, sacrificing the model`s generalization ability, which limits their applicability in real-world scenarios. To address this, we propose a novel Adaptive PET eXpert meta-learning (APEX) approach for CSP. First, SQL syntax guides the LLM to assist experts in adaptively warming up, ensuring better model initialization. Then, a dynamically expanding expert pool stores knowledge and explores the relationship between experts and instances. Finally, a selection/fusion inference strategy based on sample historical visibility promotes expert collaboration. Experiments on two CSP benchmarks show that our method achieves superior performance without data replay or ideal settings, effectively handling cold start scenarios and generalizing to unseen tasks, even surpassing performance upper bounds."
}
