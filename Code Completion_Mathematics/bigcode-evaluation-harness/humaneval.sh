#!/bin/bash
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