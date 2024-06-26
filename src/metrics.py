import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead


def perplexity(data, model, tokenizer):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

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
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            if isinstance(model, AutoModelForCausalLMWithValueHead):
                neg_log_likelihood = outputs[1] * trg_len
            else:
                neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

    return torch.exp(torch.stack(nlls).sum() / end_loc).item()


class LearnedRewardMetric:
    def __init__(
        self,
        RewardModel,
        label_ix,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(RewardModel)
        self.tokenizer.truncation_side = "left"

        self.model = AutoModelForSequenceClassification.from_pretrained(RewardModel).to(
            self.device
        )

        self.label_ix = label_ix
        self.batch_size = 100
        

    def compute(self, data):
        all_scores = []
        current_ix = 0
        n_texts = len(data)
        
        while current_ix < n_texts:

            batch = data[current_ix : current_ix + self.batch_size]

            encoded = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=encoded.input_ids.to(self.device),
                    attention_mask=encoded.attention_mask.to(self.device),
                )
                scores = torch.softmax(outputs.logits, dim=1)
                scores = scores[:, self.label_ix].tolist()
                all_scores.extend(scores)
            current_ix += self.batch_size

        return list(map(torch.tensor, all_scores))