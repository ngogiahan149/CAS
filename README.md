# CAS
Leverage Large Language Models for Implicit Constrained Data Augmentation with Semantic Enrichment in Biomedical Relation Extraction
## Biomedical Relation Extraction
### Fine-tuning RE model
```
python script.py train --mode biore --train_file dataset/train_test_aug2.csv --test_file test_transformers.csv
```
### Fine-tuning Novelty model
```
python script.py train --mode novelty --train_file dataset/train_original_large.csv --test_file dataset/test.csv
```
### Inference of both RE and Novelty
```
python script.py evaluate --eval_file dataset/bc8_biored_task1_test.csv --path_relation model_all --path_novel model_novelty_no_none_novel --output_file result/bc8_biored_task1_test-relation-no_none_novel-modelall.csv
```
Details performance in the result/result_leaderboard.txt file is calculated by the Leaderboard of BioCreative VIII BioRED Track Subtask 1 Challenge
## Code Completion 
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
## Mathematics Reasoning
## Information Retrieval
