

import os
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args

class GradientNormAnalyzer:
    def __init__(self):
        self.norm_history = defaultdict(list)
        self.global_norm_history = []
        self.parameter_stats = {}
        
    def compute_parameter_norms(self, state_dict, prefix=""):
        """Compute Frobenius norm for each parameter"""
        norms = {}
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                norm = torch.norm(param, p='fro').item()
                norms[f"{prefix}{name}"] = norm
        return norms
    
    def compute_global_norm(self, state_dict):
        """Compute global norm across all parameters"""
        total_norm = 0.0
        for param in state_dict.values():
            if isinstance(param, torch.Tensor):
                total_norm += torch.norm(param, p='fro').item() ** 2
        return np.sqrt(total_norm)
    
    def analyze_gradient_update(self, global_dict, local_dict, round_num, client_id):
        """Analyze the gradient update (difference between local and global)"""
        gradient_dict = {}
        for name in global_dict.keys():
            if name in local_dict:
                gradient_dict[name] = local_dict[name] - global_dict[name]
        
        # Compute norms
        param_norms = self.compute_parameter_norms(gradient_dict, f"R{round_num}_C{client_id}_")
        global_norm = self.compute_global_norm(gradient_dict)
        
        # Store in history
        for name, norm in param_norms.items():
            self.norm_history[name].append(norm)
        self.global_norm_history.append({
            'round': round_num,
            'client': client_id,
            'global_norm': global_norm,
            'param_norms': param_norms
        })
        
        return param_norms, global_norm
    
    def get_statistics(self):
        """Compute statistics for all tracked parameters"""
        stats = {}
        
        # Global norm statistics
        global_norms = [entry['global_norm'] for entry in self.global_norm_history]
        if global_norms:
            stats['global_norm'] = {
                'mean': np.mean(global_norms),
                'std': np.std(global_norms),
                'min': np.min(global_norms),
                'max': np.max(global_norms),
                'median': np.median(global_norms),
                'p95': np.percentile(global_norms, 95),
                'p99': np.percentile(global_norms, 99)
            }
        
        # Per-parameter statistics
        stats['parameter_stats'] = {}
        for param_name, norms in self.norm_history.items():
            if norms:
                stats['parameter_stats'][param_name] = {
                    'mean': np.mean(norms),
                    'std': np.std(norms),
                    'min': np.min(norms),
                    'max': np.max(norms),
                    'median': np.median(norms),
                    'p95': np.percentile(norms, 95),
                    'p99': np.percentile(norms, 99)
                }
        
        return stats
    
    def recommend_clipping_thresholds(self, stats, privacy_level="moderate"):
        """Recommend clipping thresholds based on observed norms"""
        recommendations = {}
        
        global_stats = stats.get('global_norm', {})
        if global_stats:
            if privacy_level == "conservative":
                # Clip at median to preserve most gradients
                rec_global = global_stats['median']
            elif privacy_level == "moderate":
                # Clip at 95th percentile
                rec_global = global_stats['p95']
            else:  # aggressive
                # Clip at 99th percentile
                rec_global = global_stats['p99']
            
            recommendations['global_clip_norm'] = rec_global
        
        # Per-parameter recommendations
        param_recommendations = {}
        for param_name, param_stats in stats.get('parameter_stats', {}).items():
            if privacy_level == "conservative":
                param_recommendations[param_name] = param_stats['median']
            elif privacy_level == "moderate":
                param_recommendations[param_name] = param_stats['p95']
            else:
                param_recommendations[param_name] = param_stats['p99']
        
        recommendations['parameter_clip_norms'] = param_recommendations
        
        return recommendations
    
    def save_analysis(self, output_dir):
        """Save analysis results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save statistics
        stats = self.get_statistics()
        with open(os.path.join(output_dir, 'gradient_norm_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save recommendations
        recommendations = {}
        for level in ['conservative', 'moderate', 'aggressive']:
            recommendations[level] = self.recommend_clipping_thresholds(stats, level)
        
        with open(os.path.join(output_dir, 'clipping_recommendations.json'), 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        # Create plots
        self.create_plots(output_dir, stats)
        
        return stats, recommendations
    
    def create_plots(self, output_dir, stats):
        """Create visualization plots"""
        # Global norm distribution
        global_norms = [entry['global_norm'] for entry in self.global_norm_history]
        if global_norms:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.hist(global_norms, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Global Gradient Norm')
            plt.ylabel('Frequency')
            plt.title('Distribution of Global Gradient Norms')
            plt.axvline(np.median(global_norms), color='red', linestyle='--', label='Median')
            plt.axvline(np.percentile(global_norms, 95), color='orange', linestyle='--', label='95th Percentile')
            plt.legend()
            
            plt.subplot(2, 2, 2)
            rounds = [entry['round'] for entry in self.global_norm_history]
            plt.plot(rounds, global_norms, 'o-', alpha=0.7)
            plt.xlabel('Training Round')
            plt.ylabel('Global Gradient Norm')
            plt.title('Global Gradient Norm Over Time')
            
            # Top parameter norms
            plt.subplot(2, 2, 3)
            param_means = [(name, np.mean(norms)) for name, norms in self.norm_history.items()]
            param_means.sort(key=lambda x: x[1], reverse=True)
            top_params = param_means[:10]  # Top 10 parameters
            
            names, means = zip(*top_params) if top_params else ([], [])
            plt.barh(range(len(names)), means)
            plt.yticks(range(len(names)), [name.split('_')[-1] for name in names])
            plt.xlabel('Mean Gradient Norm')
            plt.title('Top 10 Parameters by Mean Gradient Norm')
            
            # Norm evolution over time for top parameters
            plt.subplot(2, 2, 4)
            for name, _ in top_params[:5]:  # Top 5 parameters
                rounds_for_param = []
                norms_for_param = []
                for entry in self.global_norm_history:
                    if name in entry['param_norms']:
                        rounds_for_param.append(entry['round'])
                        norms_for_param.append(entry['param_norms'][name])
                
                if rounds_for_param:
                    plt.plot(rounds_for_param, norms_for_param, 'o-', 
                           label=name.split('_')[-1], alpha=0.7)
            
            plt.xlabel('Training Round')
            plt.ylabel('Parameter Gradient Norm')
            plt.title('Parameter Gradient Norms Over Time')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'gradient_norm_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()

def main():
    print("🔍 Starting Gradient Norm Analysis for DP-FedLoRA...")
    
    # ===== Configuration =====
    script_args, fed_args, peft_config = get_config()
    training_args = get_training_args(script_args, script_args.learning_rate)
    
    # Override some settings for analysis
    fed_args.num_rounds = min(10, fed_args.num_rounds)  # Analyze first 10 rounds
    script_args.use_dp = False  # Disable DP for norm analysis
    
    print(f"📊 Analyzing {fed_args.num_rounds} rounds with {fed_args.num_clients} clients")
    
    # ===== Load and process dataset =====
    dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
    dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)
    local_datasets = split_dataset(fed_args, script_args, dataset)
    sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]
    
    # ===== Load model =====
    device_map, quantization_config, torch_dtype = get_model_config(script_args)
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
    )
    
    if script_args.load_in_8bit or script_args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )
    
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    
    # ===== Setup tokenizer and data collator =====
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    
    # ===== Initialize analyzer =====
    analyzer = GradientNormAnalyzer()
    
    # ===== Setup federated learning =====
    global_dict = copy.deepcopy(get_peft_model_state_dict(model))
    local_dict_list = [copy.deepcopy(global_dict) for _ in range(fed_args.num_clients)]
    proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
    global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)
    
    # ===== Training loop with norm analysis =====
    training_loss = [[] for _ in range(fed_args.num_clients)]
    
    for round in tqdm(range(fed_args.num_rounds), desc="Analyzing Rounds"):
        clients_this_round = get_clients_this_round(fed_args, round)
        print(f"\n>> === Round {round+1}: Analyzing Clients {clients_this_round} ===")
        
        for client in range(fed_args.num_clients):
            if client not in clients_this_round:
                training_loss[client].append(-1)
                continue
            
            # Store initial global state
            global_state_before = copy.deepcopy(global_dict)
            
            set_peft_model_state_dict(model, global_dict)
            sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)
            new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)
            training_args = get_training_args(script_args, new_lr)
            
            trainer = get_fed_local_sft_trainer(
                model=model,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=sub_dataset,
                formatting_prompts_func=formatting_prompts_func,
                data_collator=data_collator,
                global_dict=global_dict,
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=auxiliary_model_list[client],
                global_auxiliary=global_auxiliary,
            )
            
            results = trainer.train()
            training_loss[client].append(results.training_loss)
            
            # Get local update and analyze
            local_state = get_peft_model_state_dict(model)
            
            # Analyze gradient norms
            param_norms, global_norm = analyzer.analyze_gradient_update(
                global_state_before, local_state, round, client
            )
            
            print(f"   Client {client}: Global norm = {global_norm:.6f}")
            print(f"   Top 3 param norms: {sorted(param_norms.items(), key=lambda x: x[1], reverse=True)[:3]}")
            
            local_dict_list[client] = copy.deepcopy(local_state)
            
            if fed_args.fed_alg == 'scaffold':
                auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()
        
        # Global aggregation
        global_dict, global_auxiliary = global_aggregate(
            fed_args, global_dict, local_dict_list, sample_num_list,
            clients_this_round, round,
            proxy_dict=proxy_dict,
            opt_proxy_dict=opt_proxy_dict,
            auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
        )
        
        set_peft_model_state_dict(model, global_dict)
        torch.cuda.empty_cache()
    
    # ===== Save analysis results =====
    output_dir = os.path.join(script_args.output_dir, "gradient_norm_analysis")
    stats, recommendations = analyzer.save_analysis(output_dir)
    
    # ===== Print recommendations =====
    print("\n" + "="*60)
    print("🎯 GRADIENT NORM ANALYSIS COMPLETE!")
    print("="*60)
    
    if 'global_norm' in stats:
        global_stats = stats['global_norm']
        print(f"\n📊 Global Gradient Norm Statistics:")
        print(f"   Mean: {global_stats['mean']:.6f}")
        print(f"   Std:  {global_stats['std']:.6f}")
        print(f"   Min:  {global_stats['min']:.6f}")
        print(f"   Max:  {global_stats['max']:.6f}")
        print(f"   Median: {global_stats['median']:.6f}")
        print(f"   95th percentile: {global_stats['p95']:.6f}")
        print(f"   99th percentile: {global_stats['p99']:.6f}")
    
    print(f"\n🔧 Recommended Clipping Thresholds:")
    for level in ['conservative', 'moderate', 'aggressive']:
        if level in recommendations:
            global_clip = recommendations[level].get('global_clip_norm', 'N/A')
            print(f"   {level.capitalize()}: dp_clip_norm={global_clip:.4f}")
    
    print(f"\n📁 Detailed analysis saved to: {output_dir}")
    print(f"   📊 Statistics: gradient_norm_stats.json")
    print(f"   🔧 Recommendations: clipping_recommendations.json")
    print(f"   📈 Plots: gradient_norm_analysis.png")
    
    print(f"\n💡 Usage in your DP training script:")
    moderate_clip = recommendations.get('moderate', {}).get('global_clip_norm', 1.0)
    print(f"   dp_clip_norm={moderate_clip:.4f}")

if __name__ == "__main__":
    main()