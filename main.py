import os
import yaml
import argparse
from pprint import pprint
from peft import LoraConfig

from trl.core import set_seed
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from src.utils import *


#####################################################################
# Parsing Args
parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Config File')
parser.add_argument('-og', '--only_generate', help='If you only want to generate', action='store_true')
args = parser.parse_args()
#####################################################################


#####################################################################
# Reading Config File
with open(args.config, "r") as f:
    config = yaml.safe_load(f)
pprint(config, indent=4, width=2)
print()
#####################################################################


#####################################################################
set_seed(config['Seed'])
#####################################################################


#####################################################################
# Importing Data
data = build_data(config['Dataset'], config.get('DataConfig'))
print('\n', data)  
#####################################################################


#####################################################################
# Supervised Fine Tuning
if config['Task'] == 'SFT':

    # Default Model Location
    model_dir = os.path.join(config['Model'], 'default')

    # Creating Lora Config
    if config.get('LoraConfig'):
        peft_config = LoraConfig(**config['LoraConfig'])
    else:
        peft_config = None

    # Creating SFT Config & Trainer
    sft_config = SFTConfig(
        output_dir = config['OutputDir'],
        dataset_text_field = 'reference',
        **config.get('SFTConfig', {})
        )
    trainer = SFTTrainer(
        model = model_dir,
        args = sft_config,
        train_dataset = data['train'],
        eval_dataset = data['eval'],
        peft_config=peft_config
        )

    
    if not args.only_generate:
        # Training
        trainer.train()

        # Saving Best Model
        output_dir = os.path.join(config['OutputDir'], config['Task'], 'default')
        trainer.save_model(output_dir)


    tokenizer = trainer.tokenizer
    model = trainer.model

    # Creating Prompts and Generating Completions
    data['test'] = create_prompts(data['test'], tokenizer, config['PromptConfig'])
    data['test'] = compute_generations(data['test'], model, tokenizer, config['GenerationConfig'])
#####################################################################


#####################################################################
# Reinforcement Learning from Human Feedback
elif config['Task'] == 'RLHF':
    # tokenizer = build_tokenizer(model_dir, config['TokenizerConfig'])
    # model = build_model(model_dir, tokenizer, config.get('LoraConfig'))
    pass
#####################################################################


#####################################################################
# Direct Preference Optimization
elif config['Task'] == 'DPO':
    pass
#####################################################################


#####################################################################
# Computing Metrics
config['MetricConfig']['RewardModel'] = os.path.join(config['MetricConfig']['RewardModel'], 'default')
metrics = compute_metrics(data['test'], model, tokenizer, config['MetricConfig'])
pprint(metrics)

if not args.only_generate:
    trainer.log(metrics)
#####################################################################