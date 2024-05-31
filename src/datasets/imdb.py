import os
from datasets import load_dataset


def dataloader(config):
    
    def preprocess(sample):
        text = sample['text']
        sample['reference'] = text
        return sample

    dataset = load_dataset("imdb")
    dataset.pop('unsupervised')
    dataset = dataset.map(preprocess)
    dataset.shuffle()

    if config.get('only_positive', False):
        # Keeping only Positive Sentiment
        dataset['train'] = dataset["train"].filter(lambda sample: sample["label"] == 1)

    # Train Eval Split
    train_eval = dataset['train'].train_test_split(test_size=config['eval_ratio'], stratify_by_column="label")

    dataset['train'] = train_eval['train']
    dataset['eval'] = train_eval['test']

    dataset['test'] = dataset['test'].select(range(config['test_size']))

    dataset = dataset.remove_columns(["text", "label"])

    return dataset