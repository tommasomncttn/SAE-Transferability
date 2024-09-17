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

    # SAE configs
    SAE_RELEASE : str
    LAYER_NUM : int 
    HOOK_PART : str

    # misc
    DTYPE: torch.dtype = torch.float16
    IS_DATASET_TOKENIZED: bool = False

    # sizes for experiments
    SUBSTITUTION_LOSS_BATCH_SIZE: int = 25
    L0_LOSS_BATCH_SIZE: int = 50
    FEATURE_ACTS_BATCH_SIZE: int = 25
    FEATURE_DENSITY_BATCH_SIZE: int = 50

    # parameters for the activation store
    STORE_BATCH_SIZE_PROMPTS: int = 8
    TRAIN_BATCH_SIZE_TOKENS: int = 4096
    N_BATCHES_IN_BUFFER: int = 32
    N_BATCH_TOKENS: int = None # will be computed later

class Experiment(Enum):
    SUBSTITUTION_LOSS = 'SubstitutionLoss'
    L0_LOSS = 'L0_loss'
    FEATURE_ACTS = 'FeatureActs'
    FEATURE_DENSITY = 'FeatureDensity'

def get_sae_id_and_layer(cfg: ScoresConfig):
    layer_num, hook_part = cfg.LAYER_NUM, cfg.HOOK_PART
    return f'blocks.{layer_num}.hook_resid_{hook_part}', layer_num

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


def compute_score(model, sae, experiment: Experiment, batch_size_prompts, total_batches_dict, cfg: ScoresConfig,
                  activation_store=None, tokens_path=''):
    try:
        all_tokens = torch.load(tokens_path)
        base_model_run = False

        def get_tokens(k):
            """Returns the tokens for the k-th outer batch, where 0 <= k < TOTAL_BATCHES"""
            start_idx = k * batch_size_prompts
            end_idx = (k + 1) * batch_size_prompts

            # Get the corresponding batch of tokens from all_tokens
            tokens = all_tokens[start_idx:end_idx]  # [N_BATCH, N_CONTEXT]
            return tokens
        
    except FileNotFoundError:
        assert activation_store is not None, 'Activation store must be passed when running this function for the 1st time (i.e. for the base model)'
        base_model_run = True

        def get_tokens(k):
            """Returns the tokens sampled from the activation store"""
            # Get the corresponding batch of tokens from all_tokens
            tokens = activation_store.get_batch_tokens()  # [N_BATCH, N_CONTEXT]
            return tokens
    
    def get_batch_size(key: Experiment):
        return total_batches_dict[key]
    
    total_batches = get_batch_size(experiment)
    if base_model_run:
        all_tokens = []

    score_function = experiment_to_function(experiment)
    *scores, tokens_dataset = score_function(model, sae, total_batches, get_tokens, base_model_run, cfg)

    if base_model_run:
        torch.save(tokens_dataset, tokens_path)

    return scores

def experiment_to_function(experiment: Experiment):
    """All function that compute score must have the following signature:
        (model, sae, total_batches, get_tokens, base_model_run, cfg: ScoresConfig) -> *, Optional[tokens_sample]
    """
    if experiment == Experiment.L0_LOSS:
        return get_L0_loss
    elif experiment == Experiment.SUBSTITUTION_LOSS:
        return get_substitution_and_reconstruction_losses
    elif experiment == Experiment.FEATURE_ACTS:
        return get_feature_activations
    elif experiment == Experiment.FEATURE_DENSITY:
        return get_feature_densities

def get_L0_loss(model, sae, total_batches, get_tokens, base_model_run, cfg: ScoresConfig):
    sae_id, layer_num = get_sae_id_and_layer(cfg)
    if base_model_run:
        all_tokens = []

    all_L0 = []

    for k in tqdm(range(total_batches)):
        # Get a batch of tokens from the dataset
        tokens = get_tokens(k)  # [N_BATCH, N_CONTEXT]

        # Store tokens for later reuse
        if base_model_run:
            all_tokens.append(tokens)

        # Run the model and store the activations
        _, cache = model.run_with_cache(tokens, stop_at_layer=layer_num + 1, \
                                             names_filter=[sae_id])  # [N_BATCH, N_CONTEXT, D_MODEL]

        # Get the activations from the cache at the sae_id
        original_activations = cache[sae_id]

        # Encode the activations with the SAE
        feature_activations = sae.encode_standard(original_activations) # the result of the encode method of the sae on the "sae_id" activations (a specific activation tensor of the LLM)
        feature_activations.to('cpu')

        # Store the encoded activations
        all_L0.append(L0_loss(feature_activations))

        # Explicitly free up memory by deleting the cache and emptying the CUDA cache
        del cache
        del original_activations
        del feature_activations
        torch.cuda.empty_cache()

    tokens_dataset = torch.cat(all_tokens) if base_model_run else None
    l0_loss = torch.tensor(all_L0).mean()

    return l0_loss, tokens_dataset

def get_substitution_and_reconstruction_losses(model, sae, total_batches, get_tokens, base_model_run, cfg: ScoresConfig):
    sae_id, layer_num = get_sae_id_and_layer(cfg)
    if base_model_run:
        all_tokens = []

    tokens_dataset = torch.cat(all_tokens) if base_model_run else None
    return None, tokens_dataset

def get_feature_activations(model, sae, total_batches, get_tokens, base_model_run, cfg: ScoresConfig):
    sae_id, layer_num = get_sae_id_and_layer(cfg)
    if base_model_run:
        all_tokens = []

    tokens_dataset = torch.cat(all_tokens) if base_model_run else None
    return None, tokens_dataset

def get_feature_densities(model, sae, total_batches, get_tokens, base_model_run, cfg: ScoresConfig):
    sae_id, layer_num = get_sae_id_and_layer(cfg)
    if base_model_run:
        all_tokens = []

    tokens_dataset = torch.cat(all_tokens) if base_model_run else None
    return None, tokens_dataset

### Main function
def compute_scores(cfg: ScoresConfig):
    # get some info for experiments and some functions
    TOTAL_BATCHES_DICT = {
        Experiment.SUBSTITUTION_LOSS: cfg.SUBSTITUTION_LOSS_BATCH_SIZE,
        Experiment.L0_LOSS: cfg.L0_LOSS_BATCH_SIZE,
        Experiment.FEATURE_ACTS: cfg.FEATURE_ACTS_BATCH_SIZE,
        Experiment.FEATURE_DENSITY: cfg.FEATURE_DENSITY_BATCH_SIZE
    }

    # Define the saving names
    saving_name_base = cfg.BASE_MODEL if "/" not in cfg.BASE_MODEL else cfg.BASE_MODEL.split("/")[-1]
    saving_name_ft = cfg.FINETUNE_MODEL if "/" not in cfg.FINETUNE_MODEL else cfg.FINETUNE_MODEL.split("/")[-1]
    saving_name_ds = cfg.DATASET_NAME if "/" not in cfg.DATASET_NAME else cfg.DATASET_NAME.split("/")[-1]

    # load the base model
    device = get_device()
    base_model = HookedSAETransformer.from_pretrained(cfg.BASE_MODEL, device=device, dtype=cfg.DTYPE)

    # define the sae_id and the import the SAE
    sae_id, _ = get_sae_id_and_layer(cfg)
    sae, cfg_dict, sparsity = SAE.from_pretrained(
                            release = cfg.SAE_RELEASE,
                            sae_id = sae_id,
                            device = device
    )
    assert(cfg_dict["activation_fn_str"] == "relu")

    # get the activations store
    activation_store = ActivationsStore.from_sae(
        model=base_model,
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
    cfg.N_BATCH_TOKENS = batch_size_tokens

    # setup the logger
    _, datapath = get_env_var()
    log_path = datapath / 'log'
    logger = setup_logger(log_path, f'sae_scores_{saving_name_base}_vs_{saving_name_ft}')

    ## BASE model ##
    logger.info('SAE scores on the BASE model:')

    ### L0 loss ###
    l0_loss_tokens_path = datapath / f'L0_loss_tokens_{saving_name_base}.pt'
    l0_loss = compute_score(base_model, sae, Experiment.L0_LOSS, batch_size_prompts, TOTAL_BATCHES_DICT, cfg,
                            activation_store=activation_store, tokens_path=l0_loss_tokens_path)
    logger.info(f'L0 loss = {l0_loss[0].item()}')

    ## Finetune model ##
    # Offload the base model
    del base_model, activation_store
    clear_cache()

    # Load the finetune model
    finetune_model_hf = AutoModelForCausalLM.from_pretrained(cfg.FINETUNE_MODEL)
    finetune_model = HookedSAETransformer.from_pretrained(cfg.BASE_MODEL, device=device, 
                                                          hf_model=finetune_model_hf, dtype=cfg.DTYPE)
    del finetune_model_hf
    clear_cache()

    logger.info('SAE scores on the FINETUNE model:')
    ### L0 loss ###
    l0_loss = compute_score(finetune_model, sae, Experiment.L0_LOSS, batch_size_prompts, TOTAL_BATCHES_DICT, cfg,
                            tokens_path=l0_loss_tokens_path)
    logger.info(f'L0 loss = {l0_loss[0].item()}')


    # Ensure the log is flushed
    for handler in logger.handlers:
        handler.flush()