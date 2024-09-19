# Standard imports
import os
import torch
import numpy as np
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import einops
from utils import *
from datasets import load_dataset
from sae_lens import SAE, HookedSAETransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sae_lens import LanguageModelSAERunnerConfig
from sae_lens import ActivationsStore
import os
from dotenv import load_dotenv
import typing
from dataclasses import dataclass
from tqdm import tqdm
import logging
import torch 
import torch.nn.functional as F
from utils import *
from enum import Enum

# GPU memory saver (this script doesn't need gradients computation)
torch.set_grad_enabled(False)

class Similarity(Enum):
    COSINE = 'Cosine_Similarity'
    EUCLEDIAN_DISTANCE = 'L2_distance'

@dataclass
class SimilaritiesConfig:
    BASE_MODEL: str
    FINETUNE_MODEL: str
    FINETUNE_DATASET_NAME: str
    BASE_ACTIVATIONS: torch.Tensor
    FINETUNE_ACTIVATIONS: torch.Tensor

def compute_similarities_of_activations(cfg: SimilaritiesConfig):
    # get some info for the paths and devices
    device = get_device()
    pythonpath, datapath = get_env_var()
    saving_name_base = cfg.BASE_MODEL if "/" not in cfg.BASE_MODEL else cfg.BASE_MODEL.split("/")[-1]
    saving_name_ft = cfg.FINETUNE_MODEL if "/" not in cfg.FINETUNE_MODEL else cfg.FINETUNE_MODEL.split("/")[-1]
    saving_name_ds = cfg.FINETUNE_DATASET_NAME if "/" not in cfg.FINETUNE_DATASET_NAME else cfg.FINETUNE_DATASET_NAME.split("/")[-1]

    # store the activations
    base_activations = cfg.BASE_ACTIVATIONS.to(device)
    finetune_activations = cfg.FINETUNE_ACTIVATIONS.to(device)

    # create the saving paths
    saving_path_euclidean = datapath / f"euclidean_similarity_{saving_name_base}_{saving_name_ft}_on_{saving_name_ds}.pt"
    saving_path_cosine = datapath / f"cosine_similarity_{saving_name_base}_{saving_name_ft}_on_{saving_name_ds}.pt"
    
    # Call the functions to compute cosine similarity and Euclidean distance
    cosine_similarity = compute_cosine_similarity(base_activations, finetune_activations)
    euclidean_distance = compute_euclidean_distance(base_activations, finetune_activations)

    # Save the results
    torch.save(cosine_similarity, saving_path_cosine)
    torch.save(euclidean_distance, saving_path_euclidean)

    return {
        Similarity.COSINE: cosine_similarity,
        Similarity.EUCLEDIAN_DISTANCE: euclidean_distance
    }

# 1. Plot heatmap of similarity metric
def plot_similarity_heatmap(ST):
    fig = px.imshow(ST.cpu().numpy(), color_continuous_scale='Viridis',
                    labels={'x': 'Context (Tokens)', 'y': 'Batch Index'},
                    title="Similarity Heatmap (Batches vs Context)")
    fig.show()

# 2. Reduce ST across context dimension to [N_BATCH] and plot histogram across batches
def plot_batch_histogram(ST):
    batch_mean_similarity = torch.mean(ST, dim=1).cpu().numpy()  # Shape [N_BATCH]
    fig = px.histogram(batch_mean_similarity, nbins=30, labels={'value': 'Mean Similarity'},
                       title="Histogram of Mean Similarity Across Batches")
    fig.show()

# 3. Flatten ST into [N_BATCH * N_CONTEXT] and plot histogram across all tokens
def plot_token_histogram(ST):
    flattened_ST = ST.flatten().cpu().numpy()  # Shape [N_BATCH * N_CONTEXT]
    fig = px.histogram(flattened_ST, nbins=100, labels={'value': 'Similarity'},
                       title="Histogram of Similarity Across All Tokens")
    fig.show()

# 4. Reduce ST across batch dimension to [N_CONTEXT] and plot line plot across context
def plot_context_line(ST, N_CONTEXT):
    context_mean_similarity = torch.mean(ST, dim=0).cpu().numpy()  # Shape [N_CONTEXT]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(N_CONTEXT)), y=context_mean_similarity,
                             mode='lines', name='Context Mean Similarity'))
    fig.update_layout(title="Line Plot of Mean Similarity Across Context",
                      xaxis_title="Context (Tokens)", yaxis_title="Mean Similarity")
    fig.show()

# 5. Report the global mean value of the similarity metric
def report_global_mean(ST):
    global_mean_similarity = torch.mean(ST).item()
    print(f"Global mean similarity: {global_mean_similarity}")

    return global_mean_similarity
