
# DP-FedLoRA: Privacy-Enhanced Federated Fine-Tuning for On-Device Large Language Models

**DP-FedLoRA** is a research-oriented extension to [OpenFedLLM](https://github.com/rui-ye/OpenFedLLM), introducing **differential privacy** into federated instruction tuning of large language models (LLMs). It combines **Low-Rank Adaptation (LoRA)** with **(ε, δ)-Differential Privacy** to enable secure and efficient fine-tuning across edge devices, with minimal performance loss.

📄 **Paper**: _DP-FedLoRA: Privacy-Enhanced Federated Fine-Tuning for On-Device LLMs_  
[PDF](./papers/paper.txt)

---

## Highlights

- **Differential Privacy**: Clip + Gaussian noise on LoRA matrices with formal (ε, δ)-DP guarantees.
- **LoRA-based Efficient Tuning**: Compatible with quantized LLaMA-2 models (7B/13B).
- **Extends OpenFedLLM**: Supports all major FL algorithms (FedAvg, FedProx, SCAFFOLD, etc.).
- **Variance & Expectation Analysis**: Per-round measurement of DP impact.
- **Evaluated on 5 Benchmarks**: Close-ended tasks using standardized `lm-evaluation-harness`.

---

## Installation

```bash
git clone --recursive https://github.com/user-name/repo-name.git
cd repo-name
conda create -n dp-fedlora python=3.10
conda activate dp-fedlora
pip install -r requirements.txt
source setup.sh
```

---

## Training

DP-FedLoRA adds privacy-preserving LoRA fine-tuning support on top of OpenFedLLM’s instruction tuning framework.

### Run Training Script

```bash
bash training_scripts/DP-FedLora.sh
```

### Example CLI

```bash
CUDA_VISIBLE_DEVICES=0 python main_DPFedLoRA.py \
 --model_name_or_path "meta-llama/Llama-2-7b-hf" \
 --dataset_name "vicgalle/alpaca-gpt4" \
 --dataset_sample 20000 \
 --fed_alg "fedavg" \
 --num_clients 20 \
 --sample_clients 2 \
 --max_steps 10 \
 --num_rounds 200 \
 --batch_size 16 \
 --gradient_accumulation_steps 1 \
 --seq_length 512 \
 --peft_lora_r 32 \
 --peft_lora_alpha 64 \
 --use_peft \
 --load_in_8bit \
 --enable_dp \
 --dp_epsilon 25.0 \
 --dp_delta 1e-5 \
 --dp_clip_norm 0.1 \
 --output_dir "./output" \
 --template "alpaca"
```

---

## Differential Privacy Pipeline

Implemented in [`utils/dp_utils_stats.py`](./utils/dp_utils_stats.py):

### Gaussian Noise Injection

- Clipping via Frobenius norm
- Additive Gaussian noise to `lora_A`, `lora_B` matrices only

```python
apply_dp_to_lora_params(state_dict, clip_norm, sigma)
```

### Noise Calibration

```python
sigma = compute_sigma(epsilon, delta, clip_norm)
```

### Privacy Budget Tracking

```python
tracker = PrivacyTracker(delta, clip_norm)
tracker.add_round(sigma)
total_eps, _ = tracker.get_total_privacy()
```

---

## DP Statistics & Visualization

DP-FedLoRA provides real-time tracking of LoRA statistics:

### Measured per Round

- `||A||_F`, `||B||_F`
- Analytical variance: `Var[βA]`, `Var[Bα]`, `Var[βα]`
- Sampled means: mean_orig_update, mean_noisy_update, mean_update_diff

```python
compute_param_stats(state_dict, sigma, compute_expectation=True)
```

### Export to CSV

```bash
./output/lora_stats/dp_stats_round_{r}.csv
```

### Variance Plots

```bash
./output/expected_value_and_variance_rank_{r}.png
```

```python
plot_statistics(stat_per_round, output_dir, lora_rank)
```

---

## Benchmarks & Evaluation

DP-FedLoRA is evaluated on diverse close-ended tasks using [**lm-evaluation-harness**](https://github.com/EleutherAI/lm-evaluation-harness):

| Benchmark   | Focus                           | Eval Type         |
|-------------|----------------------------------|-------------------|
| **MMLU**     | General factual knowledge        | Multiple-choice   |
| **BBH**      | Logical reasoning (Big-Bench Hard) | Multiple-choice |
| **CRASS**    | Counterfactual commonsense       | Multiple-choice   |
| **DROP**     | Reading comprehension            | QA (free-form)    |
| **HumanEval**| Python code generation           | Pass@k            |

All benchmarks are evaluated using unified prompting and compatible with federated and DP settings. Results are discussed in detail in the [paper](./DP-FedLora.pdf).

---

## Built On

- ✅ [OpenFedLLM (KDD 2024)](https://github.com/rui-ye/OpenFedLLM)
- 🤗 [Transformers](https://github.com/huggingface/transformers) + [PEFT](https://github.com/huggingface/peft)
- 🧪 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- 📊 Matplotlib + pandas for stats tracking

---


## 🧠 Acknowledgements

We thank the maintainers of OpenFedLLM, HuggingFace, PEFT, and lm-eval-harness for the foundational libraries that made this project possible.

---