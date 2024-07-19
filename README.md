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
    - GPT2:
        - Perplexity:  34.95
        - Reward Model Score:  0.51
    - GPT2-large:
        - Perplexity:  23.62
        - Reward Model Score:  0.49
1. SFT
    - GPT2:
        - Perplexity:  36.46
        - Reward Model Score:  0.66
    - GPT2-large:
        - Perplexity:  23.81
        - Reward Model Score:  0.64
1. RLHF
    - GPT2_SFT: 
        - Perplexity:  42.6
        - Reward Model Score:  0.95
1. DPO
    - GPT2_SFT: 
        - Perplexity:  42.05
        - Reward Model Score:  0.95
    - GPT2-large:
        - Perplexity:  29.8
        - Reward Model Score:  0.98