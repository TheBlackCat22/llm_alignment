import os
from datasets import load_dataset


def dataloader(config):

    dataset = load_dataset("imdb")
    dataset.pop('unsupervised')

    test_eval = dataset['test'].train_test_split(test_size=config['test_size'], stratify_by_column = 'label')
    dataset['test'] = test_eval['test']
    dataset['eval'] = test_eval['train']

    if config.get('only_positive', False):
        # Keeping only Positive Sentiment
        dataset['train'] = dataset["train"].filter(lambda sample: sample["label"] == 1)
        dataset['eval'] = dataset["eval"].filter(lambda sample: sample["label"] == 1)

    dataset['eval'] = dataset['eval'].select(range(config['eval_size']))

    return dataset