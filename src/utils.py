import os
from tqdm import tqdm
import torch
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from trl.core import LengthSampler
from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead

import src.metrics as metrics


def build_data(dataset, data_config):

    if data_config is None:
        data_config = {}

    if dataset == 'imdb':
        from src.datasets.imdb import dataloader
        return dataloader(data_config)
    
    else:
        pass


def create_prompts(dataset, tokenizer, prompt_config):

    input_size = LengthSampler(prompt_config['min_tokens'], prompt_config['max_tokens'])

    def tokenize(sample):
        tokens = tokenizer.encode(sample['text'], padding=False, truncation=False)
        prompt_tokens = tokens[: input_size()]
        sample['input_ids'] = prompt_tokens
        sample['query'] = tokenizer.decode(prompt_tokens)
        return sample

    dataset = dataset.map(tokenize)
    dataset.set_format(type="torch")

    return dataset


def compute_generations(data, model, tokenizer, generation_config):

    generation_config = GenerationConfig(pad_token_id = tokenizer.eos_token_id, **generation_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    batch_size = 16
    n_prompts = len(data['query'])
    current_ix = 0
    generated_texts = []
    while current_ix < n_prompts:

        prompt_texts = data['query'][current_ix : current_ix + batch_size]
        prompt_tokens = tokenizer(prompt_texts, return_tensors="pt", padding=True, return_length=False).to(device)

        with torch.no_grad():
            gen_tokens = model.generate(**prompt_tokens, generation_config = generation_config)

        gen_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        generated_texts.extend(gen_texts)

        current_ix += batch_size

    data = data.add_column('response', generated_texts)
    
    return data


def compute_metrics(data, model, tokenizer, metric_config):

    perplexity = metrics.perplexity(data, model, tokenizer)

    rewards = metrics.LearnedRewardMetric(metric_config['RewardModel'], metric_config['label_idx']).compute(data['response'])
    lrm_score = torch.mean(rewards).item()

    return {
        'perplexity' : perplexity,
        'LearnedModelScore' : lrm_score
    }


def build_tokenizer(tokenizer_path):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def build_policy_model(model_dir, tokenizer, lora_config):
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_dir, peft_config=lora_config)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


def ppo_trainer_train(trainer, generation_config, reward_model):

    highest_mean_reward = -float('inf')

    for _epoch, batch in tqdm(enumerate(trainer.dataloader)):
        query_tensors = batch["input_ids"]

        # Get response from gpt2
        response_tensors, ref_response_tensors = trainer.generate(query_tensors, return_prompt=False, generate_ref_response=True, pad_token_id = trainer.tokenizer.eos_token_id, **generation_config)
        batch["response"] = trainer.tokenizer.batch_decode(response_tensors)
        batch["ref_response"] = trainer.tokenizer.batch_decode(ref_response_tensors)

        # Compute sentiment score of model output
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        rewards = reward_model.compute(texts)

        # Compute sentiment score of reference model output
        ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
        ref_rewards = reward_model.compute(ref_texts)
        batch["ref_rewards"] = ref_rewards

        # Run PPO step
        stats = trainer.step(query_tensors, response_tensors, rewards)
        trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])

        rewards = torch.tensor(rewards)
        if rewards.mean() > highest_mean_reward:
            trainer._save_pretrained(os.path.join(trainer.config.project_kwargs['logging_dir'], 'best_model'))
            highest_mean_reward = rewards.mean()
        
    return trainer