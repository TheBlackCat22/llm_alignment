# Alignment of LLMs

## Problems:
### 1. Imdb Positive Sentiment Review Generation
**Problem Statement**: Align pre-trained llms to positively generate imdb rewiews based on a given prompt.
**Solutions**:
1. Base Pre Trained Model
    - LLM: GPT2 
    - Scores:
        - Perplexity:  36.40
        - Reward Model Score:  0.61
1. SFT
    - LLM: GPT2
    - Scores:
        - Perplexity:  39.16
        - Reward Model Score:  0.67
2. RLHF
    - LLM: GPT2_SFT 
    - Scores: 
        - Perplexity:  163.80
        - Reward Model Score:  0.98