#!/bin/bash
accelerate launch main.py \
  --model codellama/CodeLlama-7b-Instruct-hf \
  --max_length_generation 650 \
  --tasks multiple-sh \
  --temperature 0.2 \
  --n_samples 1 \
  --batch_size 5 \
  --load_in_4bit \
  --load_generations_path output/code/codellamainstruct7-output-sh.json \
  --allow_code_execution