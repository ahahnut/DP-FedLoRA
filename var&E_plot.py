import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# === Load All Files ===
folder_path = "./DP_LORA_STATS/fedavg_Rank32/alpaca-gpt4_20000_fedavg_c20s2_i10_b16a1_l512_r32a128_20250602153836/lora_stats/"
file_pattern = os.path.join(folder_path, "dp_stats_round_*.csv")
all_files = sorted(glob.glob(file_pattern))

dfs = []
for file in all_files:
    round_num = int(os.path.basename(file).split("_")[-1].split(".")[0])
    df = pd.read_csv(file)
    df['round'] = round_num
    dfs.append(df)

if not dfs:
    raise ValueError("No files matched.")

# Combine
full_df = pd.concat(dfs, ignore_index=True)

# Simplify layer names
full_df['layer_short'] = full_df['layer'].apply(
    lambda x: x.split("layers.")[-1].replace(".self_attn.", "_").replace("_proj_stats", "")
)

# === Plot Total Variance over Rounds ===
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=full_df,
    x='round',
    y='Total_Var',
    hue='layer_short',
    marker='o'
)
plt.title("Total Variance over Epochs per Layer")
plt.xlabel("Epoch (Round)")
plt.ylabel("Total Variance")
plt.grid(True)
plt.legend(title="Layer", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# === Plot Expected Value (mean_noisy_update) over Rounds ===
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=full_df,
    x='round',
    y='mean_noisy_update',
    hue='layer_short',
    marker='o'
)
plt.title("Expected Value (mean_noisy_update) over Epochs per Layer")
plt.xlabel("Epoch (Round)")
plt.ylabel("Expected Value")
plt.grid(True)
plt.legend(title="Layer", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

sns.lineplot(data=full_df, x='round', y='Var_betaA', label='Var_betaA')
sns.lineplot(data=full_df, x='round', y='Var_Balpha', label='Var_Balpha')
sns.lineplot(data=full_df, x='round', y='Var_ba', label='Var_ba')
plt.title("Variance Components over Epochs")
plt.xlabel("Epoch (Round)")
plt.ylabel("Variance Value")
plt.legend()
plt.grid(True)
plt.show()

