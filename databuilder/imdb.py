from datasets import load_dataset
from transformers import AutoTokenizer

from trl.core import LengthSampler

def build_dataset(model_name, data_config):
    """
    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(data_config['query_dataset'], split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)  # Maybe needs to be removed

    input_size = LengthSampler(data_config['input_min_text_length'], data_config['input_max_text_length'])

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(
            sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}