import pandas as pd
import os
import glob

# === Configuration ===
folder_path = "../OpenFedLLM/server_trained_files/DP_LORA_STATS/meta-llama/Llama-2-13b-hf_fedavg_Rank32/alpaca-gpt4_20000_fedavg_c20s2_i10_b16a1_l512_r32a128_20250602172935/lora_stats/"
file_pattern = os.path.join(folder_path, "dp_stats_round_*.csv")

# === Load Files ===
all_files = sorted(glob.glob(file_pattern))
dfs = []

for file in all_files:
    round_num = int(os.path.basename(file).split("_")[-1].split(".")[0])
    df = pd.read_csv(file)
    df['round'] = round_num
    dfs.append(df)

# === Combine All Data ===
full_df = pd.concat(dfs, ignore_index=True)

# === Group by Round to Compute Averages ===
summary_df = full_df.groupby("round")[["mean_noisy_update", "Total_Var"]].mean().reset_index()

# === Take the Last N Rounds (e.g., Final 5 Rounds) ===
last_n = 5
convergence_window = summary_df.tail(last_n)

# === Compute Mean and Std for Convergence Estimates ===
convergence_stats = convergence_window.agg({
    "mean_noisy_update": ["mean", "std"],
    "Total_Var": ["mean", "std"]
})

# === Print Results ===
print(f"\nConvergence Statistics from Final {last_n} Rounds:\n")
print(convergence_stats)
