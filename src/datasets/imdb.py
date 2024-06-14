import os
from datasets import load_dataset


def dataloader(config, seed):

    dataset = load_dataset("imdb")
    dataset.pop('unsupervised')

    dataset.shuffle(seed = seed)

    dataset = dataset.filter(lambda sample: len(sample["text"]) > 200)

    dataset['eval'] = dataset['test'].select(range(config['test_size'], len(dataset['test'])))
    dataset['test'] = dataset['test'].select(range(0, config['test_size']))

    if config.get('only_positive', False):
        # Keeping only Positive Sentiment
        dataset['train'] = dataset["train"].filter(lambda sample: sample["label"] == 1)
        dataset['eval'] = dataset["eval"].filter(lambda sample: sample["label"] == 1)

    dataset['eval'] = dataset['eval'].select(range(config['eval_size']))

    return dataset