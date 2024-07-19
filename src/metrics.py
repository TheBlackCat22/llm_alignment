import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead


def perplexity(data, model, tokenizer):

    encodings = tokenizer("\n\n".join(data["text"]), return_tensors="pt")

    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    for i in range(0, seq_len, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len)
        trg_len = end_loc - i  # may be different from stride on last loop

        # get inputs and target ids
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to('cuda')
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            if isinstance(model, AutoModelForCausalLMWithValueHead):
                neg_log_likelihood = outputs[1] * trg_len
            else:
                neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

    return torch.exp(torch.stack(nlls).sum() / end_loc).item()


class LearnedRewardMetric:
    def __init__(
        self,
        RewardModel,
        label_ix,
    ):

        self.tokenizer = AutoTokenizer.from_pretrained(RewardModel)
        self.tokenizer.truncation_side = "left"

        self.model = AutoModelForSequenceClassification.from_pretrained(RewardModel, device_map = 'cuda')

        self.label_ix = label_ix
        self.batch_size = 100
        

    def compute(self, data):
        all_scores = []
        current_ix = 0
        n_texts = len(data)
        
        while current_ix < n_texts:

            batch = data[current_ix : current_ix + self.batch_size]

            encoded = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True).to('cuda')

            with torch.no_grad():
                outputs = self.model(
                    input_ids=encoded.input_ids,
                    attention_mask=encoded.attention_mask,
                )
                scores = torch.softmax(outputs.logits, dim=1)
                scores = scores[:, self.label_ix].tolist()
                all_scores.extend(scores)
            current_ix += self.batch_size

        return list(map(torch.tensor, all_scores))


def kl_div(data, model, ref_model, tokenizer):
    
    def logprobs_from_logits(logits, response):
        all_logprob = torch.nn.functional.log_softmax(logits, dim=-1)
        logprob = torch.gather(all_logprob, 2, torch.unsqueeze(response, -1)).squeeze(-1)
        return logprob

    current_ix = 0
    batch_size = 32

    kls = []
    while current_ix < len(data):

        batch = data[current_ix : current_ix + batch_size]

        query_len = tokenizer(batch['query'], return_tensors="pt", truncation=True, padding=True).input_ids.shape[1]
        tokens = tokenizer(batch['response'], return_tensors="pt", truncation=True, padding=True).to('cuda')

        with torch.no_grad():

            logit = model(**tokens).logits 
            ref_logit = ref_model(**tokens).logits 

            logprob = logprobs_from_logits(logit[:, query_len:-1], tokens['input_ids'][:, query_len+1:])
            ref_logprob = logprobs_from_logits(ref_logit[:, query_len:-1], tokens['input_ids'][:, query_len+1:])

            kl = (logprob - ref_logprob).sum(1)
            kls.append(kl)

        current_ix += batch_size

    kls = torch.cat(kls, 0)
    
    return kls.mean().item()