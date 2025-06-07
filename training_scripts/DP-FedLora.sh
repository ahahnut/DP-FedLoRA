#!/bin/bash

# ===== DP-FedLoRA Training Configuration =====
max_steps=10
num_rounds=200
batch_size=16
gradient_accumulation_steps=1
seq_length=512
num_clients=20
sample_clients=2
lora_r=8
lora_alpha=128
lr=5e-5

# Dataset and Model Configuration
dataset_name="vicgalle/alpaca-gpt4" #data_path or name
dataset_sample=20000
model_name_or_path="meta-llama/Llama-2-7b-hf" #hf-model or path

# Hardware Configuration
gpu=0 #mention the GPU that you use

# Federated Learning Algorithm
fed_alg="fedavg"  # Options: fedavg, scaffold, fedprox, fedyogi, fedavgm, fedadam, fedadagrad.

# ===== Differential Privacy Settings =====
dp_epsilon=25.0
dp_delta=1e-5
dp_clip_norm=0.1
use_dp=true  # Set to false to disable DP


#we support only alpaca template currently


# Create dynamic output directory
output_dir="DP_LORA_STATS/${fed_alg}_Rank${lora_r}"
mkdir -p "$output_dir"

echo "🚀 Starting DP-FedLoRA Training..."
echo "🔐 Privacy Parameters: ε=$dp_epsilon, δ=$dp_delta, Clip Norm=$dp_clip_norm"
echo "📦 Dataset: $dataset_name, Model: $model_name_or_path"
echo "💾 Output Directory: $output_dir"


# Run the DP-FedLoRA Training Script
CUDA_VISIBLE_DEVICES=$gpu python main_dp_stats.py \
    --learning_rate $lr \
    --model_name_or_path $model_name_or_path \
    --dataset_name $dataset_name \
    --dataset_sample $dataset_sample \
    --fed_alg $fed_alg \
    --num_clients $num_clients \
    --sample_clients $sample_clients \
    --num_rounds $num_rounds \
    --max_steps $max_steps \
    --batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --seq_length $seq_length \
    --peft_lora_r $lora_r \
    --peft_lora_alpha $lora_alpha \
    --use_peft \
    --load_in_8bit \
    --output_dir $output_dir \
    --template "alpaca" \
    --use_dp $use_dp \
    --dp_epsilon $dp_epsilon \
    --dp_delta $dp_delta \
    --dp_clip_norm $dp_clip_norm
    

echo "✅ DP-FedLoRA Training completed!"
echo "📁 Results saved to: $output_dir"
echo "📉 Training loss: $output_dir/training_loss.npy"
echo "🔐 Privacy log: $output_dir/privacy_budget.txt"