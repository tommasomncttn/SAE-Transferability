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
from torcheval.metrics import R2Score

# Must-have for forward-pass only scripts
torch.set_grad_enabled(False)


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

    # optional argument to load stuff offline
    HF_MODEL_BASE = None
    HF_MODEL_FINETUNE = None
    DATASET_PATH = None

class Experiment(Enum):
    SUBSTITUTION_LOSS = 'SubstitutionLoss'
    L0_LOSS = 'L0_loss'
    FEATURE_ACTS = 'FeatureActs'
    FEATURE_DENSITY = 'FeatureDensity'

def get_sae_id_and_layer(cfg: ScoresConfig):
    layer_num, hook_part = cfg.LAYER_NUM, cfg.HOOK_PART
    return f'blocks.{layer_num}.hook_resid_{hook_part}', layer_num


def compute_score(model, sae, experiment: Experiment, batch_size_prompts, total_batches_dict, cfg: ScoresConfig,
                  activation_store=None, tokens_path=''):
    """Function that performs a single experiment. Depending on the `experiment` parameter passed, it will call one of the functions 
    defined in experiment_to_function() with sampling function that samples directly from the activation_store (if the function
    is run for the base model), or with the stored tokens dataset (loaded using the tokens_path).
    """
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

    clear_cache()
    return scores

def experiment_to_function(experiment: Experiment):
    """All function that compute a score must have the following signature:
        (model, sae, total_batches, get_tokens, base_model_run, cfg: ScoresConfig) -> *, Optional[tokens_sample_tensor]
    """
    if experiment == Experiment.L0_LOSS:
        return get_L0_loss
    elif experiment == Experiment.SUBSTITUTION_LOSS:
        return get_substitution_and_reconstruction_losses
    elif experiment == Experiment.FEATURE_ACTS:
        return get_feature_activations
    elif experiment == Experiment.FEATURE_DENSITY:
        return get_feature_densities
    else:
        raise ValueError(f'Unknown experiment passed: {experiment}')

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
    
    all_SL_clean = []
    all_SL_reconstructed = []
    sae_reconstruction_metric = R2Score().to(model.cfg.device)

    for k in tqdm(range(total_batches)):
        # Get a batch of tokens from the dataset
        tokens = get_tokens(k)  # [N_BATCH, N_CONTEXT]

        if base_model_run:
            # Store tokens for later reuse
            all_tokens.append(tokens)

        # Store loss
        clean_loss, reconstructed_loss = get_substitution_loss(tokens, model, sae, sae_id, sae_reconstruction_metric)

        all_SL_clean.append(clean_loss)
        all_SL_reconstructed.append(reconstructed_loss)

    clean_loss, reconstructed_loss = torch.tensor(all_SL_clean).mean().item(), torch.tensor(all_SL_reconstructed).mean().item()
    recontruction_score = sae_reconstruction_metric.compute().item()

    tokens_dataset = torch.cat(all_tokens) if base_model_run else None
    return clean_loss, reconstructed_loss, recontruction_score, tokens_dataset

def get_feature_activations(model, sae, total_batches, get_tokens, base_model_run, cfg: ScoresConfig):
    sae_id, layer_num = get_sae_id_and_layer(cfg)
    if base_model_run:
        all_tokens = []

    all_feature_acts = []

    for k in tqdm(range(total_batches)):
        # Get a batch of tokens from the dataset
        tokens = get_tokens(k)  # [N_BATCH, N_CONTEXT]
        if base_model_run:
            all_tokens.append(tokens)

        # Run the model and store the activations
        _, cache = model.run_with_cache(tokens, stop_at_layer=layer_num + 1, \
                                        names_filter=[sae_id])  # [N_BATCH, N_CONTEXT, D_MODEL]

        # Get the activations from the cache at the sae_id
        original_activations = cache[sae_id]  # [N_BATCH, N_CONTEXT, D_SAE]

        # Encode the activations with the SAE
        feature_activations = sae.encode_standard(original_activations) # the result of the encode method of the sae on the "sae_id" activations (a specific activation tensor of the LLM)
        feature_activations = feature_activations.flatten(0, 1).to('cpu')

        # Store the encoded activations
        all_feature_acts.append(feature_activations)

        # Explicitly free up memory by deleting the cache and emptying the CUDA cache
        del cache
        del original_activations
        del feature_activations
        clear_cache()

    tokens_dataset = torch.cat(all_tokens) if base_model_run else None
    feature_activations = torch.cat(all_feature_acts)

    return feature_activations, tokens_dataset

def get_feature_densities(model, sae, total_batches, get_tokens, base_model_run, cfg: ScoresConfig):
    """
    Note that this experiment could be combined with Experiment.FEATURE_ACTS, but we run it separately 
    because of the different total_batches batch size. Experiment.FEATURE_ACTS needs to store all the feature activations
    to plot the activations histogram, while this one only needs to update the densities plot.
    """
    sae_id, layer_num = get_sae_id_and_layer(cfg)
    if base_model_run:
        all_tokens = []

    total_tokens = total_batches * cfg.N_BATCH_TOKENS
    n_features = sae.cfg.d_sae

    density_plotter = FeatureDensityPlotter(n_features, total_tokens)

    for k in tqdm(range(total_batches)):
        tokens = get_tokens(k)  # [N_BATCH, N_CONTEXT]
        if base_model_run:
            all_tokens.append(tokens)

        # Run the model and store the activations
        _, cache = model.run_with_cache(tokens, stop_at_layer=layer_num + 1, \
                                        names_filter=[sae_id])  # [N_BATCH, N_CONTEXT, D_MODEL]

        # Get the activations from the cache and convert to float32 for more accurate density computation
        original_activations = cache[sae_id].float()  # [N_BATCH, N_CONTEXT, D_SAE]

        # Encode the activations with the SAE
        feature_activations = sae.encode_standard(original_activations) # the result of the encode method of the sae on the "sae_id" activations (a specific activation tensor of the LLM)
        feature_activations = feature_activations.flatten(0, 1).to('cpu')

        # Update the density histogram data
        density_plotter.update(feature_activations)

        # Explicitly free up memory by deleting the cache and emptying the CUDA cache
        del cache
        del original_activations
        del feature_activations
        clear_cache()

    tokens_dataset = torch.cat(all_tokens) if base_model_run else None
    feature_densities = density_plotter.feature_densities

    return feature_densities, tokens_dataset

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
    if cfg.HF_MODEL is not None:
        hf_model_base = AutoModelForCausalLM.from_pretrained(cfg.HF_MODEL_BASE)
    base_model = HookedSAETransformer.from_pretrained(cfg.BASE_MODEL, device=device, dtype=cfg.DTYPE, hf_model = hf_model_base)

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
        dataset=hf_dataset
    )

    # compute the sizes for the experiments
    batch_size_prompts = activation_store.store_batch_size_prompts
    batch_size_tokens = activation_store.context_size * batch_size_prompts
    cfg.N_BATCH_TOKENS = batch_size_tokens

    # setup the logger
    _, datapath = get_env_var()
    log_path = datapath / 'log'
    logger = setup_logger(log_path, f'sae_scores_{saving_name_base}_vs_{saving_name_ft}')

    ################
    ## BASE model ##
    ################
    report_memory('After Model & activation store setup:')
    logger.info('SAE scores on the BASE model:')

    ### L0 loss ###
    l0_loss_tokens_path = datapath / f'L0_loss_tokens_{saving_name_base}.pt'
    l0_loss = compute_score(base_model, sae, Experiment.L0_LOSS, batch_size_prompts, TOTAL_BATCHES_DICT, cfg,
                            activation_store=activation_store, tokens_path=l0_loss_tokens_path)
    logger.info(f'L0 loss = {l0_loss[0].item()}')   

    report_memory('After L0 loss')

    ## Substitution loss & recontruction metric ###
    sl_loss_tokens_path = datapath / f'Substitution_loss_tokens_{saving_name_base}.pt'
    scores = compute_score(base_model, sae, Experiment.SUBSTITUTION_LOSS, batch_size_prompts, TOTAL_BATCHES_DICT, cfg,
                           activation_store=activation_store, tokens_path=sl_loss_tokens_path)
    
    clean_loss, substitution_loss, recontruction_score = scores
    logger.info(f'Clean loss = {clean_loss}')
    logger.info(f'Substitution loss = {substitution_loss}')
    logger.info(f'Varience explained by SAE = {recontruction_score}')

    report_memory('After SL loss')

    ## Features activations ##
    feature_acts_tokens_path = datapath / f'Feature_acts_tokens_{saving_name_base}.pt'
    scores = compute_score(base_model, sae, Experiment.FEATURE_ACTS, batch_size_prompts, TOTAL_BATCHES_DICT, cfg,
                           activation_store=activation_store, tokens_path=feature_acts_tokens_path)
    feature_acts = scores[0]

    feature_acts_path = datapath / f'Feature_acts_{saving_name_base}_on_{saving_name_ds}.pt'
    torch.save(feature_acts, feature_acts_path)

    del feature_acts
    clear_cache()
    report_memory('After Feature acts')

    ## Features density ##
    feature_densities_tokens_path = datapath / f'Feature_densities_tokens_{saving_name_base}.pt'
    scores = compute_score(base_model, sae, Experiment.FEATURE_DENSITY, batch_size_prompts, TOTAL_BATCHES_DICT, cfg,
                           activation_store=activation_store, tokens_path=feature_densities_tokens_path)
    feature_densities = scores[0]

    feature_densities_path = datapath / f'Feature_densities_{saving_name_base}_on_{saving_name_ds}.pt'
    torch.save(feature_densities, feature_densities_path)

    del feature_densities
    clear_cache()
    report_memory('After Feature densities')

    ## Switching to the finetune model

    # Offload the base model
    del base_model, activation_store
    clear_cache()
    report_memory('After Base model offload')

    # Load the finetune model
    FINETUNE_MODEL = cfg.FINETUNE_MODEL if cfg.HF_MODEL_FINETUNE is None else cfg.HF_MODEL_FINETUNE
    finetune_model_hf = AutoModelForCausalLM.from_pretrained(FINETUNE_MODEL)
    finetune_model = HookedSAETransformer.from_pretrained(cfg.BASE_MODEL, device=device, 
                                                          hf_model=finetune_model_hf, dtype=cfg.DTYPE)
    del finetune_model_hf
    clear_cache()
    report_memory('After loading the finetune model')

    ###################
    # FINETUNE model ##
    ###################
    logger.info('SAE scores on the FINETUNE model:')

    ## L0 loss ###
    l0_loss = compute_score(finetune_model, sae, Experiment.L0_LOSS, batch_size_prompts, TOTAL_BATCHES_DICT, cfg,
                            tokens_path=l0_loss_tokens_path)
    logger.info(f'L0 loss = {l0_loss[0].item()}')

    ### Substitution loss & recontruction metric ###
    scores = compute_score(finetune_model, sae, Experiment.SUBSTITUTION_LOSS, batch_size_prompts, TOTAL_BATCHES_DICT, cfg,
                           tokens_path=sl_loss_tokens_path)
    
    clean_loss, substitution_loss, recontruction_score = scores
    logger.info(f'Clean loss = {clean_loss}')
    logger.info(f'Substitution loss = {substitution_loss}')
    logger.info(f'Varience explained by SAE = {recontruction_score}')

    ## Features activations ##
    scores = compute_score(finetune_model, sae, Experiment.FEATURE_ACTS, batch_size_prompts, TOTAL_BATCHES_DICT, cfg,
                           tokens_path=feature_acts_tokens_path)
    feature_acts = scores[0]

    feature_acts_path = datapath / f'Feature_acts_{saving_name_ft}_on_{saving_name_ds}.pt'
    torch.save(feature_acts, feature_acts_path)

    del feature_acts
    clear_cache()

    ## Features density ##
    scores = compute_score(finetune_model, sae, Experiment.FEATURE_DENSITY, batch_size_prompts, TOTAL_BATCHES_DICT, cfg,
                           tokens_path=feature_densities_tokens_path)
    feature_densities = scores[0]

    feature_densities_path = datapath / f'Feature_densities_{saving_name_ft}_on_{saving_name_ds}.pt'
    torch.save(feature_densities, feature_densities_path)

    del feature_densities
    clear_cache()

    # Ensure the log is flushed in the end
    for handler in logger.handlers:
        handler.flush()

if __name__ == "__main__":
    ### set this args with argparse, now hardcoded
    GEMMA=False

    if GEMMA == True:
        N_CONTEXT = 1024 # number of context tokens to consider
        N_BATCHES = 8 # number of batches to consider
        TOTAL_BATCHES = 20 

        RELEASE = 'gemma-2b-res-jb'
        BASE_MODEL = "google/gemma-2b"
        FINETUNE_MODEL = 'shahdishank/gemma-2b-it-finetune-python-codes'
        DATASET_NAME = "ctigges/openwebtext-gemma-1024-cl"
        hook_part = 'post'
        layer_num = 6
    else:
        N_CONTEXT = 128 # number of context tokens to consider
        N_BATCHES = 8 # number of batches to consider
        TOTAL_BATCHES = 100 

        RELEASE = 'gpt2-small-res-jb'
        BASE_MODEL = "gpt2-small"
        FINETUNE_MODEL = 'pierreguillou/gpt2-small-portuguese'
        DATASET_NAME = "Skylion007/openwebtext"
        hook_part = 'pre'
        layer_num = 6

    SAE_HOOK = f'blocks.{layer_num}.hook_resid_{hook_part}'

    cfg = ScoresConfig(BASE_MODEL, FINETUNE_MODEL, DATASET_NAME, RELEASE, layer_num, hook_part,
                       SUBSTITUTION_LOSS_BATCH_SIZE = 25,
                       L0_LOSS_BATCH_SIZE = 50,
                       FEATURE_ACTS_BATCH_SIZE = 25,
                       FEATURE_DENSITY_BATCH_SIZE = 50)

    compute_scores(cfg)