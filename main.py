import os
import yaml
import argparse
from pprint import pprint
from peft import LoraConfig
from datasets import load_from_disk

from trl.core import set_seed
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer 
from trl.trainer.ppo_config import PPOConfig
from trl.trainer.ppo_trainer import PPOTrainer
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.dpo_trainer import DPOTrainer
from trl.models import AutoModelForCausalLMWithValueHead

from src.utils import *
from src.metrics import LearnedRewardMetric


#####################################################################
# Parsing Args
parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Config File')
parser.add_argument('-og', '--only_generate', help='If you only want to generate', action='store_true')
args = parser.parse_args()
#####################################################################


#####################################################################
# Reading Config File
print('\n*****************')
print('Config')
print('*****************', flush=True)

with open(args.config, "r") as f:
    config = yaml.safe_load(f)
pprint(config, indent=4, width=2)
print()
#####################################################################


#####################################################################
set_seed(config['Seed'])
#####################################################################


#####################################################################
# Supervised Fine Tuning
if config['Task'] == 'SFT':

    # Importing Data
    print('\n*****************')
    print('Importing Raw Data')
    print('*****************', flush=True)
    data = build_data(config['Dataset'], config.get('DataConfig'), config['Seed']) 

    if not args.only_generate:
        
        print('\n*****************')
        print('Training')
        print('*****************', flush=True)

        # Creating Lora Config
        peft_config = LoraConfig(**config['LoraConfig']) if config.get('LoraConfig') else None

        # Creating SFT Config & Trainer
        sft_config = SFTConfig(
            output_dir = config['OutputDir'],
            dataset_text_field = 'text',
            report_to = 'tensorboard',
            save_total_limit = 1,
            save_only_model = True,
            eval_strategy = 'steps',
            load_best_model_at_end = True,
            metric_for_best_model = "eval_loss",
            **config.get('SFTConfig', {})
            )
        trainer = SFTTrainer(
            model = config['Model'],
            args = sft_config,
            train_dataset = data['train'],
            eval_dataset = data['eval'],
            peft_config=peft_config
            )

        # Training
        trainer.train()
        trainer.save_model(os.path.join(config['OutputDir'], 'best_model'))

        tokenizer = trainer.tokenizer
        model = trainer.model

    else:
        # Initializing Best Policy Model and Tokenizer
        tokenizer = build_tokenizer(os.path.join(config['OutputDir'], 'best_model'))
        model = build_policy_model(os.path.join(config['OutputDir'], 'best_model'), tokenizer)

    # Creating Prompts
    print('\n*****************')
    print('Creating Prompts')
    print('*****************', flush=True)
    data = create_prompts(data, tokenizer, config['PromptConfig'])
#####################################################################


#####################################################################
# Reinforcement Learning from Human Feedback
elif config['Task'] == 'RLHF':

    # Importing Data
    print('\n*****************')
    print('Importing Raw Data')
    print('*****************', flush=True)
    data = build_data(config['Dataset'], config.get('DataConfig'), config['Seed']) 

    # Creating Prompts
    print('\n*****************')
    print('Creating Prompts')
    print('*****************', flush=True)
    tokenizer = build_tokenizer(config['Model'])
    data = create_prompts(data, tokenizer, config['PromptConfig'])

    if not args.only_generate:
        # Creating Lora Config
        peft_config = LoraConfig(**config['LoraConfig']) if config.get('LoraConfig') else None
        
        # Initializing Policy Model
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_dir, peft_config=lora_config, device_map = 'cuda')
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        # Initializing Reward Model
        reward_model = LearnedRewardMetric(*config['RewardConfig'].values())

        # Creating PPOConfig and PPOTrainer
        ppo_config = PPOConfig(
            seed = config['Seed'],
            log_with = 'tensorboard',
            model_name = config['Model'],
            query_dataset = config['Dataset'],
            reward_model = config['RewardConfig']['RewardModel'],
            project_kwargs= {"logging_dir" : config['OutputDir']},
            tracker_project_name =  'runs',
            **config.get('PPOConfig', {})
        )
        trainer = PPOTrainer(
            config =  ppo_config,
            model = model,
            tokenizer = tokenizer,
            dataset = data['train'], 
            data_collator = collator
        )

        # Training
        print('\n*****************')
        print('Training')
        print('*****************', flush=True)
        trainer = ppo_trainer_train(trainer, config['GenerationConfig'], reward_model)

    # Initializing Best Policy Model and Tokenizer
    tokenizer = build_tokenizer(os.path.join(config['OutputDir'], 'best_model'))
    model = build_policy_model(os.path.join(config['OutputDir'], 'best_model'), tokenizer)
#####################################################################


#####################################################################
# Direct Preference Optimization
elif config['Task'] == 'DPO':

    tokenizer = build_tokenizer(config['Model'])

    if preference_data_exists(config):
        # Loading Saved Data
        print('\n*****************')
        print('Loading Saved Data')
        print('*****************', flush=True)
        print(f"  - Loading Dataset from {os.path.join(config['DatasetDir'], config['Dataset'] + '_preference')}")
        data = load_from_disk(os.path.join(config['DatasetDir'], config['Dataset'] + '_preference'))
    
    else:
        # Importing Data
        print('\n*****************')
        print('Importing Raw Data')
        print('*****************', flush=True)
        data = build_data(config['Dataset'], config.get('DataConfig'), config['Seed']) 

        # Creating Prompts
        print('\n*****************')
        print('Creating Prompts')
        print('*****************', flush=True)
        data = create_prompts(data, tokenizer, config['PromptConfig'])

        # Initializing Reward Model
        reward_model = LearnedRewardMetric(*config['RewardConfig'].values())

        # Generating Preference Data
        print('\n*****************')
        print('Generating Preference Data')
        print('*****************', flush=True)
        data = prepare_for_dpo(config, data, tokenizer, reward_model)


    if not args.only_generate:
        
        print('\n*****************')
        print('Training')
        print('*****************', flush=True)

        # Creating Lora Config
        peft_config = LoraConfig(**config['LoraConfig']) if config.get('LoraConfig') else None

        # Creating DPO Config & Trainer
        dpo_config = DPOConfig(
            output_dir = config['OutputDir'],
            report_to = 'tensorboard',
            save_total_limit = 1,
            save_only_model = True,
            eval_strategy = 'steps',
            load_best_model_at_end = True,
            metric_for_best_model = "eval_loss",
            max_length = tokenizer.model_max_length,
            max_prompt_length = config['PromptConfig']['max_tokens'],
            remove_unused_columns = False,
            **config.get('DPOConfig', {})
            )
        trainer = DPOTrainer(
            model = config['Model'],
            args = dpo_config,
            train_dataset = data['train'],
            eval_dataset = data['eval'],
            tokenizer = tokenizer,
            peft_config = peft_config
            )

        # Training
        trainer.train()
        trainer.save_model(os.path.join(config['OutputDir'], 'best_model'))

        tokenizer = trainer.tokenizer
        model = trainer.model

    else:
        # Initializing Best Policy Model and Tokenizer
        tokenizer = build_tokenizer(os.path.join(config['OutputDir'], 'best_model'))
        model = build_policy_model(os.path.join(config['OutputDir'], 'best_model'), tokenizer)   
#####################################################################


#####################################################################
# Generating Completions
print('\n*****************')
print('Generating Completions')
print('*****************', flush=True)

data['test'] = compute_generations(data['test'], model, tokenizer, config['GenerationConfig'])
#####################################################################


#####################################################################
# Computing Metrics
print('\n*****************')
print('Computing Metrics')
print('*****************', flush=True)

ref_model = build_policy_model(config['Model'], tokenizer)

metrics = compute_metrics(data['test'], model, ref_model, tokenizer, config['MetricConfig'])
pprint(metrics, indent=4, width=2)

if not args.only_generate:
    if config['Task'] == 'RLHF':
        trainer.accelerator.log(metrics, step = 0)
    else:
        trainer.log(metrics)
#####################################################################