#!/bin/bash
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
