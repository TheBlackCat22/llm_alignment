# Alignment of LLMs

## Problems:
### 1. Imdb Positive Sentiment Review Generation
**Problem Statement**: Align pre-trained llms to positively generate imdb rewiews based on a given prompt.
**Solutions**:
1. Base Pre Trained Model
    - LLM: GPT2 
    - Scores:
        - Perplexity:  34.95
        - Reward Model Score:  0.48
1. SFT
    - LLM: GPT2
    - Scores:
        - Perplexity:  36.32
        - Reward Model Score:  0.58
2. RLHF
    - LLM: GPT2_SFT 
    - Scores: 
        - Perplexity:  
        - Reward Model Score:  