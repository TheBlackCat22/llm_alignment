# Alignment of LLMs

## Problems:
### 1. Imdb Positive Sentiment Review Generation
**Problem Statement**: Align pre-trained llms to positively generate imdb rewiews based on a given prompt.
**Solutions**:
1. Base Pre Trained Model
    - LLM: GPT2 
    - Scores:
        - Perplexity:  34.96
        - Reward Model Score:  0.42
1. SFT
    - LLM: GPT2
    - Scores:
        - Perplexity:  37.7
        - Reward Model Score:  0.50
2. RLHF
    - LLM: GPT2_SFT 
    - Scores: 
        - Perplexity:
        - Reward Model Score: