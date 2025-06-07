import numpy as np

# Load the training loss data
loss_array = np.load("../OpenFedLLM/server_trained_files/output_dp/scaffold_epsilon25.0_clip0.1/alpaca-gpt4_20000_scaffold_c20s2_i10_b16a1_l512_r32a64_20250526083320/training_loss.npy")

# Flatten the array if it's multi-dimensional (e.g., rounds × clients)
loss_flat = loss_array.flatten()

# Filter out non-participating clients (-1)
participating_losses = loss_flat[loss_flat != -1]

# Compute statistics
expected_value = np.mean(participating_losses)
variance = np.var(participating_losses)
standard_deviation = np.std(participating_losses)

print(expected_value, variance, standard_deviation)
