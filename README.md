# Alignment of LLMs

Problem Statement: Align pre-trained llms towards user preferences.

## Toy Project
- **Dataset**: IMDB Reviews
- **LLM**: GPT2 (pretrained)
- **SFT**: Positive Rewiews from Dataset
- **Alignment Method**: RLHF 
- **RL Algorithm**: PPO
- **Reward Function**: Distilled BERT (pretrained)