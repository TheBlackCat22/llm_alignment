from tqdm import tqdm
import torch
from transformers import GenerationConfig

from trl.core import LengthSampler

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
        tokens = tokenizer.encode(sample['reference'], padding=False, truncation=False)
        prompt_tokens = tokens[: input_size()]
        sample['prompt_ids'] = prompt_tokens
        sample['prompt'] = tokenizer.decode(prompt_tokens)
        return sample

    dataset = dataset.map(tokenize)
    dataset.set_format(type="torch")

    return dataset


# TODO: Need to discuss if batch generation or not
def compute_generations(data, model, tokenizer, generation_config):

    generation_config = GenerationConfig(pad_token_id = tokenizer.eos_token_id, **generation_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    batch_size = 8
    n_prompts = len(data['prompt'])
    current_ix = 0
    generated_texts = []
    while current_ix < n_prompts:

        prompt_texts = data['prompt'][current_ix : current_ix + batch_size]
        prompt_tokens = tokenizer(prompt_texts, return_tensors="pt", padding=True, return_length=False).to(device)

        with torch.no_grad():
            gen_tokens = model.generate(**prompt_tokens, generation_config = generation_config)

        gen_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        generated_texts.extend(gen_texts)

        current_ix += batch_size

    data = data.add_column('generation', generated_texts)
    
    return data


def compute_metrics(data, model, tokenizer, metric_config):

    perplexity = metrics.perplexity(data, model, tokenizer)

    rewards = metrics.LearnedRewardMetric(metric_config['RewardModel'], metric_config['label_idx']).compute(data['generation'])
    lrm_score = sum(rewards)/len(rewards)

    return {
        'perplexity' : perplexity,
        'LearnedModelScore' : lrm_score
    }


# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import LoraConfig, get_peft_model

# def build_tokenizer(tokenizer_path, tokenizer_config):

#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token_id = tokenizer.eos_token_id

#     tokenizer.padding_side = "left"
#     tokenizer.truncation_side = "left"

#     return tokenizer


# def build_model(model_dir, tokenizer, lora_config):

#     model = AutoModelForCausalLM.from_pretrained(model_dir)
#     if model.config.pad_token_id is None:
#         model.config.pad_token_id = tokenizer.pad_token_id

#     if lora_config is None:
#         return model
    
#     lora_config = LoraConfig(**lora_config)
#     peft_model = get_peft_model(model, lora_config)

#     return peft_model
