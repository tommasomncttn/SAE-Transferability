import torch
import torch.nn.functional as F
from enum import Enum
import numpy as np
from scipy.stats import gamma
import os
from dotenv import load_dotenv
import gc
from pathlib import Path
import logging
import json

# Load environment variables from the .env file
load_dotenv()

# Access the PYTHONPATH variable
PYTHONPATH = os.getenv('PYTHONPATH')
DATAPATH = PYTHONPATH + '/data'

def get_env_var():
    # Load environment variables from the .env file
    load_dotenv()
    # Access the PYTHONPATH variable
    pythonpath = Path(os.getenv('PYTHONPATH'))
    # Print to verify
    print(f"PYTHONPATH: {pythonpath}")
    datapath = pythonpath / 'data'
    print(f"DATAPATH: {datapath}")

    return pythonpath, datapath

#### memory management stuff ####
def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
nvmlInit()

def report_memory(message=''):
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)

    print(message)
    print(f'free:\t {info.free / 10e8}')
    print(f'used:\t {info.used / 10e8}')

def get_device():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"
    return device
    
#### Logging utils ####
def setup_logger(log_dir_path: Path, log_name: str):
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)

    # Setup the file-logger
    logger = logging.getLogger(log_name)

    # Clear any existing handlers to ensure no console logging
    logger.handlers.clear()

    # Set the log level
    logger.setLevel(logging.INFO)

    # Create file handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_dir_path / log_name, encoding='utf-8')

    # Set the logging format
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add only the file handler to the logger
    logger.addHandler(file_handler)

    # Disable propagation to prevent any parent loggers from printing to the console
    logger.propagate = False

    return logger

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
    the threshold), a.k.a. L0 loss.
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
        yaxis_title="Count",
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

from transformer_lens import HookedTransformer
from functools import partial

def get_substitution_loss(tokens, model, sae, sae_hook, model_name, reconstruction_metric=None):
    '''
    Expects a tensor of input tokens of shape [N_BATCHES, N_CONTEXT].

    Returns two losses:
    1. Clean loss - loss of the normal forward pass of the model at the input tokens.
    2. Substitution loss - loss when substituting SAE reconstructions of the residual stream at the SAE layer of the model.
    '''
    # Run the model with cache to get the original activations and clean loss
    loss_clean, cache = model.run_with_cache(tokens, names_filter=[sae_hook], return_type="loss")

    # Fetch and detach the original activations
    original_activations = cache[sae_hook]

    # Apply activation filtering
    activations_filtered, filter_mask = filter_activations(original_activations, model_name=model_name, return_mask=True)
    # Shape of activations_filtered is now [valid_activations, d_model]

    # Filter the tokens using the same mask
    tokens_filtered = tokens[filter_mask].reshape(activations_filtered.shape[0]) # shape [valid_activations]

    # Get the SAE reconstructed activations
    post_reconstructed = sae.forward(activations_filtered) # shape [valid_activations, d_model]

    # Update the reconstruction quality metric, if provided
    if reconstruction_metric:
        reconstruction_metric.update(post_reconstructed.flatten().float(), activations_filtered.flatten().float())

    # Free unused variables early to save memory
    del original_activations, activations_filtered, cache
    clear_cache()

    # Hook function to substitute activations with SAE reconstructions
    def hook_function(activations, hook, new_activations):
        activations.copy_(new_activations)  # Perform in-place substitution of activations
        return activations

    # Run the model again with hooks to substitute activations at the SAE layer
    loss_reconstructed = model.run_with_hooks(
        tokens_filtered,
        return_type="loss",
        fwd_hooks=[(sae_hook, partial(hook_function, new_activations=post_reconstructed))]
    )

    # Clean up the reconstructed activations and clear memory
    del post_reconstructed
    clear_cache()

    return loss_clean, loss_reconstructed

### Outliers filtering ###

# First, we'll load a config that is filled with the values obtained after running notebooks/find_outlier_norms.ipynb 
def load_outliers_cfg():
    pythonpath, _ = get_env_var()
    OUTLIERS_CFG_PATH = pythonpath / 'saetuning' / 'cfg' / 'outlier_cfg.json'
    
    with open(OUTLIERS_CFG_PATH, 'r') as file:
        OUTLIERS_CFG = json.load(file)

    return OUTLIERS_CFG

OUTLIERS_CFG = load_outliers_cfg()

def get_norm_scalar(model_name):
    return OUTLIERS_CFG.get("norm_scalar", {}).get(model_name, None)

def get_threshold_multiplier(model_name):
    return OUTLIERS_CFG.get("threshhold_multiplier", {}).get(model_name, None)

def get_base_threshhold(model_name):
    return OUTLIERS_CFG.get("base_threshhold", {}).get(model_name, None)

# Main filtering method
def filter_activations(acts, model_name, return_mask=False):
    """
    Filters out activations based on outlier norms and returns the filtered activations.
    
    Args:
        acts (torch.Tensor): A tensor of activations with shape [BATCH, SEQ, D_MODEL].
        model_name (str): The name of the model used to determine the threshold for filtering out outlier activations.
        return_mask (bool): If True, returns the 2D boolean mask indicating which activations were retained. The mask has shape [BATCH, SEQ].
    
    Returns:
        torch.Tensor: A tensor of filtered activations with shape [N_VALID_ACTIVATIONS, D_MODEL], where N_VALID_ACTIVATIONS <= BATCH * SEQ.
        torch.Tensor (optional): A 2D boolean tensor of shape [BATCH, SEQ] representing the filtering mask, indicating whether each activation was retained (True) or filtered out (False).
    
    Notes:
        - The function removes activations identified as outliers by `is_act_outlier`. The activations that pass the filter are flattened into a tensor of shape [N_VALID_ACTIVATIONS, D_MODEL].
        - If `return_mask=True`, the function also returns a 2D boolean mask corresponding to the [BATCH, SEQ] dimensions of the original activations. This mask can be useful for tracking which activations were kept.
        - The returned filtered activations are flattened across both batch and sequence dimensions. If reshaping back to a sequence or batch structure is required, you will need to do this outside the function based on the original mask.
    """
    # Get the outlier mask
    is_outlier_mask = is_act_outlier(acts, model_name)  # [BATCH, SEQ]

    # Expand the mask to match the last dimension (D_MODEL) for correct filtering
    expanded_mask = is_outlier_mask.unsqueeze(-1).expand_as(acts)  # [BATCH, SEQ, D_MODEL]

    # Apply the mask and filter out the outlier activations
    filtered_acts = acts[~expanded_mask].reshape(-1, acts.shape[-1])  # Flatten only the valid activations, retaining D_MODEL

    if return_mask:
        # Return the 2D mask corresponding to the original [BATCH, SEQ] shape
        filter_mask = ~is_outlier_mask  # Keep it as 2D: [BATCH, SEQ]
        return filtered_acts, filter_mask
    else:
        return filtered_acts
    
# Auxilary method for getting a mask of outlier activations
def is_act_outlier(act_tensor, model_name):
    """
    Expects act_tensor of shape [*, D_MODEL]

    Returns a boolean tensor of shape [*], where for each batch position we report whether the corresponding activation
    exceeds the outlier threshold that is defined as
    
    threshold = threshold_multiplier * base_threshold, where
    base_threshold = sqrt(D_MODEL)

    Important! This threshold value is in the normalized scale, i.e. is meant to be used for activations that are scaled
    in such a way, that their average norm is equal to sqrt(D_MODEL). To do this normalization, we multiple by norm_scalar
    of the corresponding model.

    Check this blog-post for more details: https://www.lesswrong.com/posts/fmwk6qxrpW8d4jvbd/saes-usually-transfer-between-base-and-chat-models
    """
    norm_scalar = get_norm_scalar(model_name)
    threshold_multiplier = get_threshold_multiplier(model_name)
    base_threshold = get_base_threshhold(model_name)

    threshold = threshold_multiplier * base_threshold

    scaled_act = norm_scalar * act_tensor
    scaled_act_norms = torch.norm(scaled_act, dim=-1)

    return scaled_act_norms > threshold