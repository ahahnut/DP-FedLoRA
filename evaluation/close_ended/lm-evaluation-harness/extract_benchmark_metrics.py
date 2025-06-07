import json

# Define the JSON file path
json_file_path = '../../../../OpenFedLLM/server_trained_files/results/scaffold25.0clip0.1_withDP_crass_bbh_mmlu_drop_humaneval_2025-06-01T18-33-06.297854.json'

# Load the JSON file
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Define the keys and metrics to extract
metrics_to_extract = {
    'mmlu': ('acc,none', 'MMLU Accuracy'),
    'bbh': ('exact_match,get-answer', 'BBH Exact Match'),
    'drop': ('em,none', 'DROP EM'),
    'bigbench_crass_ai_multiple_choice': ('acc,none', 'CRASS Accuracy'),
    'humaneval': ('pass@1,create_test', 'HumanEval Pass@1')
}

# Extract and print the desired values
print("Extracted Metrics:\n------------------")
for key, (metric_key, label) in metrics_to_extract.items():
    value = data.get("results", {}).get(key, {}).get(metric_key, None)
    if value is not None:
        print(f"{label}: {value}")
    else:
        print(f"{label}: Not found")
