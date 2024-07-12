# Alignment of LLMs


# Setup:
```
git clone https://github.com/TheBlackCat22/llm_alignment
cd llm_alignment
conda create -n alignment_env python=3.10 -y
conda activate alignment_env
python setup.py
```


# Usage:
1. Choose which config file you want to run from the configs directory
1. For Training, 
    ```
    python main.py --config <path to config file>
    ```
1. For Testing, 
    ```
    python main.py --config <path to config file> -og
    ```

## Results:
### 1. Imdb Positive Sentiment Review Generation
**Problem Statement**: Align pre-trained llms to positively generate imdb rewiews based on a given prompt.
**Solutions**:
1. Base Pre Trained Model
    - LLM: GPT2 
    - Scores:
        - Perplexity:  34.95
        - Reward Model Score:  0.51
1. SFT
    - LLM: GPT2
    - Scores:
        - Perplexity:  36.46
        - Reward Model Score:  0.66
1. RLHF
    - LLM: GPT2_SFT 
    - Scores: 
        - Perplexity:  42.6
        - Reward Model Score:  0.95
1. DPO
    - LLM: GPT2_SFT 
    - Scores: 
        - Perplexity:  42.05
        - Reward Model Score:  0.95