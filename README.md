# SAE-Transferability

## TLDR (**Executive Summary)**

- We explore **whether Sparse Autoencoders (SAEs)** can effectively be transfered from base language models to their fine-tuned counterparts, focusing on two base models: [Gemma-2b](https://huggingface.co/google/gemma-2b) and [Mistral-7B-V0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) (we tested respectively a coding and mathematics fine-tuned version)
- In particular, we split our analysis in three steps:
    1. analysing the similarity (**Cosine and Euclidian Distance**) of the residual activations 
    2. we computed the performance (**Loss Delta, Variance Explained, and Dead Features**)  of base SAEs on the fine-tuned models
    3. we made a further step by operationalising the idea of transferability of SAE from base models to fine-tuned models by applying an [approach from Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features#phenomenology-universality) for studying features universality **through features activations vectors** (correlation between feature activation vectors across models) **and features logits similarity** (correlation between feature logit weight vector, reflecting the feature’s influence on model outputs)
- Our findings reveal that SAEs maintain their interpretability and reconstruction quality for Mistral-7B after fine-tuning, indicating successful transferability. Conversely, for Gemma-2b, SAEs perform poorly on the fine-tuned model, showing significant degradation, confirming the results in a [previous work](https://www.alignmentforum.org/posts/fmwk6qxrpW8d4jvbd/saes-usually-transfer-between-base-and-chat-models). This suggests that SAEs' transferability is model-dependent and sensitive to the fine-tuning process.
