import os
import copy
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import *
from utils.dp_utils_stats import apply_dp_to_lora_params, compute_sigma, PrivacyTracker, compute_param_stats, export_statistics_to_csv, plot_statistics
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args

def main():
    script_args, fed_args, peft_config = get_config()
    training_args = get_training_args(script_args, script_args.learning_rate)
    save_config(script_args, fed_args)

    dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
    dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)
    local_datasets = split_dataset(fed_args, script_args, dataset)
    sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

    device_map, quantization_config, torch_dtype = get_model_config(script_args)
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
    )

    if script_args.load_in_8bit or script_args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    global_dict = copy.deepcopy(get_peft_model_state_dict(model))
    local_dict_list = [copy.deepcopy(global_dict) for _ in range(fed_args.num_clients)]
    proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
    global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

    if script_args.use_dp:
        privacy_tracker = PrivacyTracker(delta=script_args.dp_delta, sensitivity=script_args.dp_clip_norm)
        stat_per_round = {}

    training_loss = [[] for _ in range(fed_args.num_clients)]

    for round in tqdm(range(fed_args.num_rounds)):
        clients_this_round = get_clients_this_round(fed_args, round)
        print(f"\n>> === Round {round+1}: Clients {clients_this_round} ===")

        for client in range(fed_args.num_clients):
            if client not in clients_this_round:
                training_loss[client].append(-1)
                continue

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

            local_state = get_peft_model_state_dict(model)

            if script_args.use_dp:
                sigma = compute_sigma(
                    epsilon=script_args.dp_epsilon,
                    delta=script_args.dp_delta,
                    sensitivity=script_args.dp_clip_norm,
                )

                param_stats = compute_param_stats(local_state, sigma=sigma, compute_expectation=True, num_samples=100)
                round_key = f"round_{round+1}_client_{client}"
                stat_per_round[round_key] = param_stats
                export_statistics_to_csv({round_key: param_stats}, script_args.output_dir +"/lora_stats/", round + 1)

                dp_state = apply_dp_to_lora_params(local_state, script_args.dp_clip_norm, sigma)
                eps_this_round = privacy_tracker.add_round(sigma)
                total_eps, delta = privacy_tracker.get_total_privacy()

                print(f"[DP-FedLoRA] Round {round+1}, Client {client} — σ={sigma:.4f}, ε={eps_this_round:.4f}, Total ε={total_eps:.4f}, δ={delta}")
                local_state = dp_state

            local_dict_list[client] = copy.deepcopy(local_state)

            if fed_args.fed_alg == 'scaffold':
                auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        global_dict, global_auxiliary = global_aggregate(
            fed_args, global_dict, local_dict_list, sample_num_list,
            clients_this_round, round,
            proxy_dict=proxy_dict,
            opt_proxy_dict=opt_proxy_dict,
            auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
        )

        set_peft_model_state_dict(model, global_dict)
        torch.cuda.empty_cache()

        if (round + 1) % fed_args.save_model_freq == 0:
            trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))

        np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))

    if script_args.use_dp:
        total_eps, delta = privacy_tracker.get_total_privacy()
        with open(os.path.join(script_args.output_dir, "privacy_budget.txt"), "w") as f:
            f.write(f"Total ε: {total_eps:.4f}, δ: {delta}\n")

        plot_statistics(stat_per_round, script_args.output_dir,peft_config.r)

if __name__ == '__main__':
    main()
