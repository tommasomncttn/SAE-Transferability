import torch
import torch.nn.functional as F
from enum import Enum
import numpy as np
from scipy.stats import gamma
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Access the PYTHONPATH variable
PYTHONPATH = os.getenv('PYTHONPATH')
DATAPATH = PYTHONPATH + '/data'


#### Enum for pretty code ####
class AggregationType(Enum):
    MEAN = 'mean'
    LAST= 'last'

class SimilarityMetric(Enum):
  COSINE = 'cosine'
  EUCLIDEAN = 'euclidean'


#### Similarity and Distance Computations ####

# 1. Compute pairwise cosine similarity between base and finetune activations
def compute_cosine_similarity(base_activations, finetune_activations):
    # Normalize activations along the activation dimension
    base_norm = F.normalize(base_activations, dim=-1)
    finetune_norm = F.normalize(finetune_activations, dim=-1)
    
    # Compute dot product along activation dimension to get cosine similarity
    cosine_similarity = torch.einsum('bca,bca->bc', base_norm, finetune_norm)  # [N_BATCH, N_CONTEXT]
    return cosine_similarity

# 2. Compute pairwise Euclidean distance between base and finetune activations
def compute_euclidean_distance(base_activations, finetune_activations):
    # Compute squared difference and sum along activation dimension
    euclidean_distance = torch.norm(base_activations - finetune_activations, dim=-1)  # [N_BATCH, N_CONTEXT]
    return euclidean_distance

#### CKA code ####

def linear_kernel(X, Y):
  """
  Compute the linear kernel (dot product) between matrices X and Y.
  """
  return torch.mm(X, Y.T)

def HSIC(K, L):
    """
    Calculate the Hilbert-Schmidt Independence Criterion (HSIC) between kernels K and L.
    """
    n = K.shape[0]  # Number of samples
    H = torch.eye(n) - (1./n) * torch.ones((n, n))

    KH = torch.mm(K, H)
    LH = torch.mm(L, H)
    return 1./((n-1)**2) * torch.trace(torch.mm(KH, LH))

def CKA(X, Y):
    """
    Calculate the Centered Kernel Alignment (CKA) for matrices X and Y.
    If no kernel is specified, the linear kernel will be used by default.
    """

    # Compute the kernel matrices for X and Y
    K = linear_kernel(X, X)
    L = linear_kernel(Y, Y)

    # Calculate HSIC values
    hsic = HSIC(K, L)
    varK = torch.sqrt(HSIC(K, K))
    varL = torch.sqrt(HSIC(L, L))

    # Return the CKA value
    return hsic / (varK * varL)

#### Quantitave SAE evaluation ####
def L0_loss(x, threshold=1e-8):
    """
    Expects a tensor x of shape [N_TOKENS, N_SAE].
    
    Returns a scalar representing the mean value of activated features (i.e. values across the N_SAE dimensions bigger than
    the threshhold), a.k.a. L0 loss.
    """
    return (x > threshold).float().sum(-1).mean()

import plotly.graph_objs as go

def plot_density(feature_acts, num_bins=100):
    """
    Expects a tensor feature_acts of shape [N_TOKENS, N_SAE].
    
    Computes the histogram using PyTorch and plots the feature density diagram with log-10 scale using Plotly.
    (see https://transformer-circuits.pub/2023/monosemantic-features#appendix-feature-density)
    """
    # Flatten the tensor
    feature_acts_flat = torch.flatten(feature_acts)

    # Compute the logarithmic transformation using PyTorch
    log_feature_acts_flat = torch.log10(torch.abs(feature_acts_flat) + 1e-10).detach().cpu()

    # Compute histogram using PyTorch
    hist_min = torch.min(log_feature_acts_flat).item()
    hist_max = torch.max(log_feature_acts_flat).item()
    hist_range = hist_max - hist_min
    bin_edges = torch.linspace(hist_min, hist_max, num_bins + 1)
    hist_counts, _ = torch.histogram(log_feature_acts_flat, bins=bin_edges)

    # Convert data to NumPy for Plotly
    bin_edges_np = bin_edges.detach().cpu().numpy()
    hist_counts_np = hist_counts.detach().cpu().numpy()

    # Prepare the Plotly plot
    fig = go.Figure(
        data=[go.Bar(
            x=bin_edges_np[:-1],  # Exclude the last bin edge
            y=hist_counts_np,
            width=hist_range / num_bins,
        )]
    )

    # Update the layout for the plot
    fig.update_layout(
        title="SAE Feature Density Diagram",
        xaxis_title="Log10 of Feature Activations",
        yaxis_title="Density",
        bargap=0.2,
        bargroupgap=0.1,
    )

    # Show the plot
    fig.show()

from transformer_lens import HookedTransformer
from functools import partial

def get_substitution_loss(tokens, model, sae, sae_layer):
    '''
    Expects a tensor of input tokens of shape [N_BATCHES, N_CONTEXT] (e.g. sampled from the Activation Store).

    Returns two losses:
    1. Clean loss - loss of the normal forward pass of the model at the input tokens
    2. Substitution loss - loss when substituting SAE reconstructions of the residual stream at the SAE layer of the model
    '''
    batch_size, seq_len = tokens.shape

    # Get the post activations from the clean run (and get the clean loss)
    loss_clean, cache = model.run_with_cache(tokens, names_filter = [sae_layer], return_type="loss")
    original_activations = cache[sae_layer]

    # Use these to get 'post_reconstructed' (for both autoencoder A and B). We need to reshape back to (batch, seq) first
    post_reconstructed = sae.forward(original_activations)
    
    # Define hook fn to replace activations with different values
    def hook_function(activations, hook, new_activations):
        activations[:] = new_activations
        return activations

    # Run the hook function in 3 different cases: sae A's reconstructions, B's reconstructions, and zero-ablation
    loss_reconstructed = model.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks=[(sae_layer, partial(hook_function, new_activations=post_reconstructed))],
    )

    return loss_clean, loss_reconstructed