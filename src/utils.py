import os
import yaml
from tqdm import tqdm
import numpy as np
import torch
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from trl.core import LengthSampler

import src.metrics as metrics


def build_data(dataset, data_config, seed):

    if data_config is None:
        data_config = {}

    if dataset == 'imdb':
        from src.datasets.imdb import dataloader
        data = dataloader(data_config, seed)
        print('\n', data) 
        return data
    
    else:
        pass


def create_prompts(data, tokenizer, prompt_config):

    input_size = LengthSampler(prompt_config['min_tokens'], prompt_config['max_tokens'])

    def tokenize(sample):
        tokens = tokenizer.encode(sample['text'], padding=False, truncation=False)
        prompt_tokens = tokens[: input_size()]
        sample['input_ids'] = prompt_tokens
        sample['query'] = tokenizer.decode(prompt_tokens)
        return sample

    data = data.map(tokenize)
    data.set_format(type="torch")

    data['train'] = data['train'].remove_columns(['text', 'label'])
    data['eval'] = data['eval'].remove_columns(['text', 'label'])
    data['test'] = data['test'].remove_columns(['label', 'input_ids'])

    print('\n', data) 

    return data


def compute_generations(data, model, tokenizer, generation_config):

    tokenizer.padding_side = "left"
    
    model.eval()

    generation_config = GenerationConfig(pad_token_id = tokenizer.eos_token_id, **generation_config)

    batch_size = 32

    def generate(sample):
        prompt_texts = sample['query']
        prompt_tokens = tokenizer(prompt_texts, return_tensors="pt", padding=True, return_length=False).to('cuda')

        with torch.no_grad():
            gen_tokens = model.generate(**prompt_tokens, generation_config = generation_config)

        sample['response'] = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        return sample

    data = data.map(generate, batched=True, batch_size = batch_size)
    
    return data


def compute_metrics(data, model, ref_model, tokenizer, metric_config):

    print('  - Calculating Perplexity')
    perplexity = metrics.perplexity(data, model, tokenizer)

    print('  - Calculating Learned Reward')
    rewards = metrics.LearnedRewardMetric(metric_config['RewardModel'], metric_config['label_idx']).compute(data['response'])
    lrm_score = (sum(rewards)/len(rewards)).item()

    print('  - Calculating KL Divergence\n')
    kl = metrics.kl_div(data, model, ref_model, tokenizer)

    return {
        'metrics/perplexity' : perplexity,
        'metrics/LearnedModelScore' : lrm_score,
        'metrics/KL_Div' : kl
    }


def build_tokenizer(tokenizer_path):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def build_policy_model(model_dir, tokenizer):
    
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map = 'cuda')
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


def ppo_trainer_train(trainer, generation_config, reward_model):

    if trainer.config.steps is None:
        trainer.config.steps = float('inf')

    least_loss = float('inf')

    for step_num, batch in tqdm(enumerate(trainer.dataloader)):
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

        if stats['ppo/loss/total'] < least_loss:
            trainer._save_pretrained(os.path.join(trainer.config.project_kwargs['logging_dir'], 'best_model'))
            least_loss = stats['ppo/loss/total']

        if step_num+1 == trainer.config.steps:
            break
        
    return trainer


def preference_data_exists(config):
    
    if os.path.exists(os.path.join(config['DatasetDir'], config['Dataset'] + '_preference', 'config.yaml')):

        with open(os.path.join(config['DatasetDir'], config['Dataset'] + '_preference', 'config.yaml'), "r") as f:
            saved_config = yaml.safe_load(f)

        data_config = {k: config[k] for k in ('Model', 'Dataset', 'DataConfig', 'PromptConfig', 'GenerationConfig')}

        if data_config == saved_config:
            return True
    
    return False


def prepare_for_dpo(config, data, tokenizer, reward_model):

    model = build_policy_model(config['Model'], tokenizer)
    
    print('\n  - Sampling Completion Pairs')
    data['train'] = sample_completion_pairs(data['train'], model, tokenizer, config['GenerationConfig'])
    data['eval'] = sample_completion_pairs(data['eval'], model, tokenizer, config['GenerationConfig'])

    print('\n  - Ranking Completions')
    data['train'] = rank_completions(data['train'], reward_model)
    data['eval'] = rank_completions(data['eval'], reward_model)

    data['train'] = data['train'].rename_column('query', 'prompt')
    data['eval'] = data['eval'].rename_column('query', 'prompt')

    data['train'] = data['train'].remove_columns(['input_ids'])
    data['eval'] = data['eval'].remove_columns(['input_ids'])

    print(f"\n  - Saving Dataset to {os.path.join(config['DatasetDir'], config['Dataset'] + '_preference')}")
    data.save_to_disk(os.path.join(config['DatasetDir'], config['Dataset'] + '_preference'))
    with open(os.path.join(config['DatasetDir'], config['Dataset'] + '_preference', 'config.yaml'), 'w') as outfile:
        yaml.dump(
            {k: config[k] for k in ('Model', 'Dataset', 'DataConfig', 'PromptConfig', 'GenerationConfig')}, 
            outfile, 
            default_flow_style=False
        )

    print('\n', data) 

    return data


def sample_completion_pairs(data, model, tokenizer, generation_config):

    tokenizer.padding_side = "left"
    
    model.eval()

    generation_config = GenerationConfig(pad_token_id = tokenizer.eos_token_id, **generation_config)

    batch_size = 32

    def generate(sample):
        prompt_texts = sample['query']
        prompt_tokens = tokenizer(prompt_texts, return_tensors="pt", padding=True, return_length=False).to('cuda')

        with torch.no_grad():
            gen_tokens1 = model.generate(**prompt_tokens, generation_config = generation_config)
            gen_tokens2 = model.generate(**prompt_tokens, generation_config = generation_config)

        gen_texts1 = tokenizer.batch_decode(gen_tokens1, skip_special_tokens=True)
        gen_texts2 = tokenizer.batch_decode(gen_tokens2, skip_special_tokens=True)

        sample['response1'] = gen_texts1
        sample['response2'] = gen_texts2

        return sample

    data = data.map(generate, batched=True, batch_size = batch_size)
    
    return data


def rank_completions(data, reward_model):

    texts = data['response1'] + data['response2']
    scores = reward_model.compute(texts)
    
    texts = np.array(texts).reshape(2, -1).T
    scores = np.array(scores).reshape(2, -1).T
    
    chosen = np.take_along_axis(texts, scores.argmax(axis=1, keepdims=True), axis=1)
    rejected = np.take_along_axis(texts, scores.argmin(axis=1, keepdims=True), axis=1)

    data = data.add_column("chosen", chosen.squeeze(1).tolist())
    data = data.add_column("rejected", rejected.squeeze(1).tolist())

    data = data.remove_columns(['response1', 'response2'])
    
    return data