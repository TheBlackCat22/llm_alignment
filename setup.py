import os
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

ModelDict = {
    "gpt2" : ['openai-community/gpt2', AutoModelForCausalLM],
    "gpt2-large" : ['openai-community/gpt2-large', AutoModelForCausalLM],
    "distilbert" : ['lvwerra/distilbert-imdb', AutoModelForSequenceClassification],
    "siebert" : ['siebert/sentiment-roberta-large-english', AutoModelForSequenceClassification]
}

ModelDir = './models/Default'

if not os.path.exists(ModelDir):
    os.makedirs(ModelDir)

for model_name in ModelDict.keys():

    if os.path.exists(os.path.join(ModelDir, model_name)):
        continue

    print(f'\nDownloading {model_name}')

    os.makedirs(os.path.join(ModelDir, model_name))

    model_link, model_class = ModelDict[model_name]
    model = model_class.from_pretrained(model_link)
    tokenizer = AutoTokenizer.from_pretrained(model_link)

    model.save_pretrained(os.path.join(ModelDir, model_name))
    tokenizer.save_pretrained(os.path.join(ModelDir, model_name))
    