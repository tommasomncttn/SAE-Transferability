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

