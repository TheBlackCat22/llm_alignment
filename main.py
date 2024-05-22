import os
import sys
import yaml
from tqdm import tqdm
import argparse
import torch
from peft import LoraConfig
from accelerate import Accelerator
from transformers import AutoTokenizer, pipeline

from trl.core import set_seed
from trl.trainer.ppo_config import PPOConfig
from trl.trainer.ppo_trainer import PPOTrainer
from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead


#####################################################################
# Parsing Args
parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Config File')
args = parser.parse_args()
#####################################################################


#####################################################################
# Reading Config File
with open(os.path.join("configs", args.config + '.yaml'), "r") as f:
    config = yaml.safe_load(f)
#####################################################################


set_seed(config['PPOConfig']['seed'])


#####################################################################
# Creating Dataloader
if config['DataConfig']['query_dataset'] == 'imdb':
    from databuilder.imdb import build_dataset, collator
    dataset = build_dataset(config['PPOConfig']['model_name'], config['DataConfig'])
else:
    print("Dataset not implemented yet")
    sys.exit(0)
#####################################################################


#####################################################################
# Building the model, the reference model, and the tokenizer.
if not config['LoraConfig']['use_peft']:
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config['PPOConfig']['model_name'])
    device_map = None
    peft_config = None
else:
    ref_model = None
    peft_config = LoraConfig(
        r=config['LoraConfig']['lora_r'],
        lora_alpha=config['LoraConfig']['lora_alpha'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    device_map = {"": Accelerator().local_process_index}

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config['PPOConfig']['model_name'],
    device_map=device_map,
    peft_config=peft_config,
)

tokenizer = AutoTokenizer.from_pretrained(config['PPOConfig']['model_name'])
tokenizer.pad_token_id = tokenizer.eos_token_id
#####################################################################


#####################################################################
# Creating PPO Trainer
ppo_trainer = PPOTrainer(PPOConfig(**config['PPOConfig']), model, ref_model, tokenizer, dataset=dataset, data_collator=collator)
#####################################################################


#####################################################################
# Preparing Reward Pipeline
task, model_name = config['RewardConfig']['reward_model'].split(":")

if ppo_trainer.accelerator.num_processes == 1:
    device = 0
else:
    device = ppo_trainer.accelerator.device
ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin

if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
    with ds_plugin.zero3_init_context_manager(enable=False):
        sentiment_pipe = pipeline(task, model=model_name, device=device)
else:
    sentiment_pipe = pipeline(task, model=model_name, device=device)

config['RewardConfig'].pop('reward_model')

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
if sentiment_pipe.tokenizer.pad_token_id is None:
    sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id
if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id
#####################################################################


#####################################################################
# Training Loop
for _epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from gpt2
    response_tensors, ref_response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, generate_ref_response=True, pad_token_id=tokenizer.eos_token_id, **config['GenerationConfig']
    )
    batch["response"] = tokenizer.batch_decode(response_tensors)
    batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **config['RewardConfig'])
    rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
    ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
    ref_pipe_outputs = sentiment_pipe(ref_texts, **config['RewardConfig'])
    ref_rewards = [torch.tensor(output[1]["score"])
                   for output in ref_pipe_outputs]
    batch["ref_rewards"] = ref_rewards

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=[
                          "query", "response", "ref_response", "ref_rewards"])
#####################################################################