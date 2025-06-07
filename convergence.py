import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# Load the dataset
folder_path = "./DP_LORA_STATS/fedavg_Rank32/alpaca-gpt4_20000_fedavg_c20s2_i10_b16a1_l512_r32a128_20250602153836/lora_stats/"
file_pattern = os.path.join(folder_path, "dp_stats_round_*.csv")
all_files = sorted(glob.glob(file_pattern))

dfs = []
for file in all_files:
    round_num = int(os.path.basename(file).split("_")[-1].split(".")[0])
    df = pd.read_csv(file)
    df['round'] = round_num
    dfs.append(df)

# Combine all data
full_df = pd.concat(dfs, ignore_index=True)

# Group by round and compute average expectation and variance values
summary_df = full_df.groupby("round")[["mean_noisy_update", "Total_Var"]].mean().reset_index()

# Take the last 5 rounds to estimate convergence
convergence_stats = summary_df.tail(5).agg({
    "mean_noisy_update": ["mean", "std"],
    "Total_Var": ["mean", "std"]
})

print(convergence_stats)
