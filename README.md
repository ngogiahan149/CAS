# Leverage Large Language Models for Implicit Constrained Data Augmentation with Semantic Enrichment in Biomedical Relation Extraction
Table of Contents
=================
* [Biomedical Relation Extraction](#biomedical-relation-extraction)
* [Code Completion](#code-completion)
* [Mathematics Reasoning](#mathematics-reasoning)
* [Information Retrieval](#information-retrieval)
# Biomedical Relation Extraction
## Fine-tuning RE model
```
python script.py train --mode biore --train_file dataset/train_test_aug2.csv --test_file test_transformers.csv
```
## Fine-tuning Novelty model
```
python script.py train --mode novelty --train_file dataset/train_original_large.csv --test_file dataset/test.csv
```
## Inference of both RE and Novelty
```
python script.py evaluate --eval_file dataset/bc8_biored_task1_test.csv --path_relation model_all --path_novel model_novelty_no_none_novel --output_file result/bc8_biored_task1_test-relation-no_none_novel-modelall.csv
```
Details performance in the result/result_leaderboard.txt file is calculated by the Leaderboard of BioCreative VIII BioRED Track Subtask 1 Challenge
# Code Completion 
## Fine-tuning process
Go to "finetune" folder
```
cd finetune
```
Fine-tune the model
```
DATA_PATH="evolinstruct_aug.json"
OUTPUT_PATH="finetuned_model"
MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-instruct"

deepspeed finetune_deepseekcoder.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 3 \
    --model_max_length 1024 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --deepspeed configs/ds_config_zero3.json \
    --bf16 True \
```
## Evaluation process
```
cd bigcode-evaluation-harness
```
### Humaneval (Python)
```
accelerate launch main.py \
  --model deepseek-ai/deepseek-coder-1.3b-instruct \
  --max_length_generation 512 \
  --tasks humaneval \
  --temperature 0.2 \
  --n_samples 1 \
  --batch_size 3 \
  --load_in_4bit \
  --save_generations \
  --save_generations_path output/code/deepseek-coder-1.3b-instruct.json \
  --metric_output_path output/metric/deepseek-coder-1.3b-instruct.json \
  --allow_code_execution
```
or 
```
bash humaneval.sh
```
### MultiPL-E (Java, JavaScript, C++, Bash, Rust)
Change the tasks multiple-[lang] with lang = ["cpp", "cs", "d", "go", "java", "jl" (Julia), "js", "lua", "php", "pl", "py", "r", "rb", "rkt", "rs", "scala", "sh", "swift", "ts"]
```
accelerate launch main.py \
  --model deepseek-ai/deepseek-coder-1.3b-instruct \
  --max_length_generation 650 \
  --tasks multiple-cpp \
  --temperature 0.2 \
  --n_samples 1 \
  --batch_size 3 \
  --load_in_4bit \
  --save_generations \
  --save_generations_path output/code/deepseek-coder-1.3b-instruct.json \
  --generation_only \
  --allow_code_execution
```
or
```
bash multipl-e.sh
```
# Mathematics Reasoning
```
#!/bin/bash
accelerate launch  main.py \
  --model checkpoint-800 \
  --max_length_generation 2048 \
  --tasks pal-gsm8k-greedy \
  --n_samples 1 \
  --batch_size 1 \
  --do_sample False \
  --metric_output_path output/metric/cp800-gsm8k.json \
  --save_generations \
  --save_generations_path output/code/cp800-gsm8k-output.json \
  --allow_code_execution
```
or 
```
bash pal-gsm8k.sh
```
# Information Retrieval
## LoRA fine-tuning the models
Dataset
* train_em_top1.jsonl: original data
* train_em_top1_aug_ori(1).jsonl: original + augmentation data

Model name
* llama_lora
* llama2_lora
* gpt2_lora
* gptj_lora
* opt_lora
```
python train_llama.py \
--train_data_path "./datasets/nq/xturing/mgen/train_em_top1_aug_ori(1).jsonl" \
--output_dir "./saved_model_xturing/llama_aug(1)" \
--num_train_epochs 3 \
--model_name "llama_lora"
```
## Inference
```
python query_llama.py \
--dataset_path "./datasets/nq/xturing/mgen/test_em_top1.jsonl" \
--output_path "./output/nq/mgen/silver-em_tuned-llama_aug(1).json" \
--model_name "llama_lora" \
--model_path "./saved_model_xturing/llama"
```
## Evaluate the performance
```
python eval.py \
--dataset_path "./datasets/nq/base/test.json" \
--predset_path "./output/nq/mgen/silver-em_tuned-llama_aug(1).json" \
--metric_name "em"
```
