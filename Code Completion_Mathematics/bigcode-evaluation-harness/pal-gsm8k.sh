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
