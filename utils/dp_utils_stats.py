
import torch
import math
import random
import numpy as np
import csv
import os
from collections import defaultdict
from typing import Dict
import matplotlib.pyplot as plt


def compute_sigma(epsilon: float, delta: float, sensitivity: float) -> float:
    if epsilon <= 0 or delta <= 0 or sensitivity <= 0:
        raise ValueError("Epsilon, delta, and sensitivity must be positive.")
    return (sensitivity * math.sqrt(2 * math.log(1.25 / delta))) / epsilon

def clip_and_add_noise(param_tensor: torch.Tensor, clip_norm: float, sigma: float) -> torch.Tensor:
    norm = torch.norm(param_tensor)
    if norm > clip_norm:
        param_tensor = param_tensor * (clip_norm / norm)
    noise = torch.normal(mean=0.0, std=sigma, size=param_tensor.shape, device=param_tensor.device)
    return param_tensor + noise

def apply_dp_to_lora_params(state_dict: Dict[str, torch.Tensor], clip_norm: float, sigma: float) -> Dict[str, torch.Tensor]:
    dp_state_dict = {}
    for name, param in state_dict.items():
        if 'lora' in name:
            dp_state_dict[name] = clip_and_add_noise(param, clip_norm, sigma)
        else:
            dp_state_dict[name] = param.clone()
    return dp_state_dict

def compute_param_stats(state_dict, sigma=0.0, compute_expectation=False, num_samples=100):
    stat_dict = {}

    for name, param in state_dict.items():
        if "lora_A" in name or "lora_B" in name:
            stat_dict[name] = {}
            stat_dict[name]["frobenius_norm"] = torch.norm(param, p='fro').item()
    paired_keys = set(k.rsplit(".lora_A.weight", 1)[0] for k in stat_dict if "lora_A" in k)
    for key in paired_keys:
        A_key = f"{key}.lora_A.weight"
        B_key = f"{key}.lora_B.weight"
        if A_key in state_dict and B_key in state_dict:
            A = state_dict[A_key]
            B = state_dict[B_key]
            A_norm = torch.norm(A, p='fro').item()
            B_norm = torch.norm(B, p='fro').item()
            m, r = B.shape
            r_, n = A.shape
            assert r == r_, f"LoRA shape mismatch: B.shape={B.shape}, A.shape={A.shape}"

            var_Balpha = m * sigma**2 * B_norm**2
            var_betaA = n * sigma**2 * A_norm**2
            var_ba = sigma**4 * m * n * r
            total_var = var_Balpha + var_betaA + var_ba

            stats = {
                "A_norm": A_norm,
                "B_norm": B_norm,
                "Var_Balpha": var_Balpha,
                "Var_betaA": var_betaA,
                "Var_ba": var_ba,
                "Total_Var": total_var,
            }

            if compute_expectation:
                mean_orig_update = 0
                mean_noisy_update = 0
                mean_update_diff = 0

                for _ in range(num_samples):
                    beta = torch.randn_like(B) * sigma
                    alpha = torch.randn_like(A) * sigma

                    noisy_B = B + beta
                    noisy_A = A + alpha

                    BA = B @ A
                    BA_noisy = noisy_B @ noisy_A
                    update_diff = BA_noisy - BA

                    mean_orig_update += BA.mean().item()
                    mean_noisy_update += BA_noisy.mean().item()
                    mean_update_diff += update_diff.mean().item()

                stats["mean_orig_update"] = mean_orig_update / num_samples
                stats["mean_noisy_update"] = mean_noisy_update / num_samples
                stats["mean_update_diff"] = mean_update_diff / num_samples

            stat_dict[f"{key}_stats"] = stats

    return stat_dict

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

def export_statistics_to_csv(stat_dict, output_dir, round_num):
    import pandas as pd
    create_folder_if_not_exists(output_dir)
    rows = []

    for round_client, layer_stats in stat_dict.items():
        for layer_name, stats in layer_stats.items():
            if "stats" in layer_name:
                row = {"round_client": round_client, "layer": layer_name}
                cleaned_stats = {k: v for k, v in stats.items() if k != "frobenius_norm"}
                row.update(cleaned_stats)
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, f"dp_stats_round_{round_num}.csv"), index=False)


def plot_statistics(stat_dict, output_dir, lora_rank):
    rounds = []
    variances = []
    expected_values = []

    for round_client, layer_stats in stat_dict.items():
        for layer_name, stats in layer_stats.items():
            if "stats" in layer_name:
                round_num = int(round_client.split('_')[1])
                rounds.append(round_num)
                variances.append(stats["Total_Var"])#total_var ho hai
                expected_values.append(stats["mean_update_diff"]) #update_diff ho hai

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, expected_values, label=r'$\mathbb{E}[\Delta]$', color="green")
    ax.plot(rounds, variances, label=r'$\mathrm{Var}[\Delta]$', color="blue")

    ax.set_xlabel("Round")
    ax.set_ylabel("Value")
    ax.set_title(rf"Expected Value $\mathbb{{E}}[\Delta]$ and Variance $\mathrm{{Var}}[\Delta]$ over Rounds (LoRA Rank = {lora_rank})")
    ax.grid(True)
    ax.legend()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"expected_value_and_variance_rank_{lora_rank}.png"))
    plt.close()

class PrivacyTracker:
    def __init__(self, delta: float, sensitivity: float):
        self.delta = delta
        self.sensitivity = sensitivity
        self.total_sigma_squared_inv = 0.0
        self.round_epsilons = []

    def add_round(self, sigma: float) -> float:
        epsilon = (self.sensitivity / sigma) * math.sqrt(2 * math.log(1.25 / self.delta))
        self.total_sigma_squared_inv += 1 / (sigma ** 2)
        self.round_epsilons.append(epsilon)
        return epsilon

    def get_total_privacy(self) -> (float, float):
        total_eps = self.sensitivity * math.sqrt(2 * self.total_sigma_squared_inv * math.log(1.25 / self.delta))
        return total_eps, self.delta