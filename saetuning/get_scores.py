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
import torch
import plotly.graph_objects as go

@dataclass
class ScoresConfig:

    # LLMs
    BASE_MODEL: str
    FINETUNE_MODEL: str

    # dataset
    DATASET_NAME: str
    IS_DATASET_TOKENIZED: bool = False

    # SAE configs
    SAE_RELEASE : str
    LAYER_NUM : int 
    HOOK_PART : str

    # misc
    DTYPE: torch.dtype = torch.float16

    # sizes for experiments
    SUBSTITUTION_LOSS_BATCH_SIZE: int = 25
    L0_LOSS_BATCH_SIZE: int = 50
    FEATURE_ACTS_BATCH_SIZE: int = 25
    FEATURE_DENSITY_BATCH_SIZE: int = 50

    # parameters for the activation store
    STORE_BATCH_SIZE_PROMPTS: int = 8
    TRAIN_BATCH_SIZE_TOKENS: int = 4096
    N_BATCHES_IN_BUFFER: int = 32

class Experiment(Enum):
    SUBSTITUTION_LOSS = 'SubstitutionLoss'
    L0_LOSS = 'L0_loss'
    FEATURE_ACTS = 'FeatureActs'
    FEATURE_DENSITY = 'FeatureDensity'

def plot_log10_hist(y_data, y_value, num_bins=100, first_bin_name = 'First bin value',
                    y_scalar=1.5, y_scale_bin=-2, log_epsilon=1e-10):
    """
    Computes the histogram using PyTorch and plots the feature density diagram with log-10 scale using Plotly.
    Y-axis is clipped to the value of the second-largest bin to prevent suppression of smaller values.
    """
    # Flatten the tensor
    y_data_flat = torch.flatten(y_data)

    # Compute the logarithmic transformation using PyTorch
    log_y_data_flat = torch.log10(torch.abs(y_data_flat) + log_epsilon).detach().cpu()

    # Compute histogram using PyTorch
    hist_min = torch.min(log_y_data_flat).item()
    hist_max = torch.max(log_y_data_flat).item()
    hist_range = hist_max - hist_min
    bin_edges = torch.linspace(hist_min, hist_max, num_bins + 1)
    hist_counts, _ = torch.histogram(log_y_data_flat, bins=bin_edges)

    # Convert data to NumPy for Plotly
    bin_edges_np = bin_edges.detach().cpu().numpy()
    hist_counts_np = hist_counts.detach().cpu().numpy()

    # Find the largest and second-largest bin values
    first_bin_value = hist_counts_np[0]
    scale_bin_value = sorted(hist_counts_np)[y_scale_bin]  # Get the second largest bin value (by default)

    # Prepare the Plotly plot
    fig = go.Figure(
        data=[go.Bar(
            x=bin_edges_np[:-1],  # Exclude the last bin edge
            y=hist_counts_np,
            width=hist_range / num_bins,
        )]
    )

    # Update the layout for the plot, clipping the y-axis at the second largest bin value
    fig.update_layout(
        title=f"SAE Features {y_value} histogram ({first_bin_name}: {first_bin_value:.2e})",
        xaxis_title=f"Log10 of {y_value}",
        yaxis_title="Density",
        yaxis_range=[0, scale_bin_value * y_scalar],  # Clipping to the second-largest value by default
        bargap=0.2,
        bargroupgap=0.1,
    )

    # Add an annotation to display the value of the first bin
    fig.add_annotation(
        text=f"{first_bin_name}: {first_bin_value:.2e}",
        xref="paper", yref="paper",
        x=0.95, y=0.95,
        showarrow=False,
        font=dict(size=12, color="red"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )

    # Show the plot
    fig.show()

class FeatureDensityPlotter:
    def __init__(self, n_features, n_tokens, activation_threshold=1e-10, num_bins=100):
        self.num_bins = num_bins
        self.activation_threshold = activation_threshold

        self.n_tokens = n_tokens
        self.n_features = n_features

        # Initialize a tensor of feature densities for all features,
        # where feature density is defined as the fraction of tokens on which the feature has a nonzero value.
        self.feature_densities = torch.zeros(n_features, dtype=torch.float32)

    def update(self, feature_acts):
        """
        Expects a tensor feature_acts of shape [N_TOKENS, N_FEATURES].

        Updates the feature_densities buffer:
        1. For each feature, count the number of tokens that the feature activated on (i.e. had an activation greater than the activation_threshold)
        2. Add this count at the feature's position in the feature_densities tensor, divided by the total number of tokens (to compute the fraction)
        """

        activating_tokens_count = (feature_acts > self.activation_threshold).float().sum(0)
        self.feature_densities += activating_tokens_count / self.n_tokens

    def plot(self, num_bins=100, y_scalar=1.5, y_scale_bin=-2, log_epsilon=1e-10):
        plot_log10_hist(self.feature_densities, 'Density', num_bins=num_bins, first_bin_name='Dead features density',
                        y_scalar=y_scalar, y_scale_bin=y_scale_bin, log_epsilon=log_epsilon)

#### Loss functions


def get_L0_loss():
    try:
        pass # try to load the file
    except:
    



### Main function

def compute_scores(cfg: ScoresConfig):

    # get some info for experiments and some functions

    TOTAL_BATCHES = {
        Experiment.SUBSTITUTION_LOSS: cfg.SUBSTITUTION_LOSS_BATCH_SIZE,
        Experiment.L0_LOSS: cfg.L0_LOSS_BATCH_SIZE,
        Experiment.FEATURE_ACTS: cfg.FEATURE_ACTS_BATCH_SIZE,
        Experiment.FEATURE_DENSITY: cfg.FEATURE_DENSITY_BATCH_SIZE
    }

    TOKENS_SAMPLE = {
        Experiment.SUBSTITUTION_LOSS: [],
        Experiment.L0_LOSS: [],
        Experiment.FEATURE_ACTS: [],
        Experiment.FEATURE_DENSITY: []
    }

    def get_batch_size(key: Experiment):
        return TOTAL_BATCHES[key]

    def get_tokens_sample(key: Experiment):
        return TOKENS_SAMPLE[key]

    def set_tokens_sample(key: Experiment, token_sample):
        TOKENS_SAMPLE[key] = token_sample

    # load the base model
    device = get_device()
    base_model = HookedSAETransformer.from_pretrained(cfg.BASE_MODEL, device=device, dtype=cfg.DTYPE)

    # define the sae_id and the import the SAE
    sae_id = f'blocks.{cfg.LAYER_NUM}.hook_resid_{cfg.HOOK_PART}'

    sae, cfg_dict, sparsity = SAE.from_pretrained(
                            release = cfg.SAE_RELEASE,
                            sae_id = sae_id,
                            device = device
    )

    assert(cfg_dict["activation_fn_str"] == "relu")

    # get the activations store
    activation_store = ActivationsStore.from_sae(
        model=cfg.BASE_MODEL,
        sae=sae,
        streaming=True,
        # fairly conservative parameters here so can use same for larger
        # models without running out of memory.
        store_batch_size_prompts=cfg.STORE_BATCH_SIZE_PROMPTS,
        train_batch_size_tokens=cfg.TRAIN_BATCH_SIZE_TOKENS,
        n_batches_in_buffer=cfg.N_BATCHES_IN_BUFFER,
        device=device,
    )

    # compute the sizes for the experiments
    batch_size_prompts = activation_store.store_batch_size_prompts
    batch_size_tokens = activation_store.context_size * batch_size_prompts

    # write a function for loss computation for the base model, which check if file exists and import or compute from scratch

    