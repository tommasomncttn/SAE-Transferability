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
    second_largest_bin_value = sorted(hist_counts_np)[y_scale_bin]  # Get the second largest bin value (by default)

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
        yaxis_range=[0, second_largest_bin_value * y_scalar],  # Clipping to the second-largest value by default
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


from transformer_lens import HookedTransformer
from functools import partial

def get_substitution_loss(tokens, model, sae, sae_layer, reconstruction_metric=None):
    '''
    Expects a tensor of input tokens of shape [N_BATCHES, N_CONTEXT].

    Returns two losses:
    1. Clean loss - loss of the normal forward pass of the model at the input tokens
    2. Substitution loss - loss when substituting SAE reconstructions of the residual stream at the SAE layer of the model
    '''
    batch_size, seq_len = tokens.shape
    
    # Run the model with cache to get the original activations and clean loss
    loss_clean, cache = model.run_with_cache(tokens, names_filter=[sae_layer], return_type="loss")

    # Fetch and detach the original activations
    original_activations = cache[sae_layer]

    # Get the SAE reconstructed activations (forward pass through SAE)
    post_reconstructed = sae.forward(original_activations)

    # Update the reconstruction quality metric (e.g. R2 score/variance explained)
    if reconstruction_metric:
        reconstruction_metric.update(post_reconstructed.flatten(), original_activations.flatten())

    # Clear the cache and unused variables early
    del original_activations, cache
    torch.cuda.empty_cache()

    # Hook function to substitute activations in-place
    def hook_function(activations, hook, new_activations):
        activations.copy_(new_activations)  # In-place copy to save memory
        return activations

    # Run model again with hooks to substitute activations and get the substitution loss
    loss_reconstructed = model.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks=[(sae_layer, partial(hook_function, new_activations=post_reconstructed))]
    )

    # Clean up reconstructed activations and free up memory
    del post_reconstructed
    torch.cuda.empty_cache()

    return loss_clean, loss_reconstructed