#!/bin/bash

# Configuration
MODEL_TYPE="hf"
PRETRAINED_MODEL="meta-llama/Llama-2-7b-hf"
PEFT_PATH="../../../server_trained_files/output_dp/fedadgrad_epsilon25.0_clip0.1/alpaca-gpt4_20000_fedadgrad_c20s2_i10_b16a1_l512_r32a64_20250527123119/checkpoint-200"
DEVICE="cuda:0"
BATCH_SIZE=1
LIMIT=100
OUTPUT_PATH="results/fedadgrad_e25.0_clip0.1_withDP_crass_bbh_mmlu_drop_humaneval.json"

# Tasks to run
TASKS="bigbench_crass_ai_multiple_choice,bbh,mmlu,drop,humaneval"

# Run evaluation
lm-eval \
  --model $MODEL_TYPE \
  --model_args pretrained=$PRETRAINED_MODEL,peft=$PEFT_PATH,load_in_8bit=True,trust_remote_code=True \
  --tasks $TASKS \
  --device $DEVICE \
  --batch_size $BATCH_SIZE \
  --limit $LIMIT \
  --output_path $OUTPUT_PATH \
  --confirm_run_unsafe_code