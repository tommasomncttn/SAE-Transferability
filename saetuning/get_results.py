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

# GPU memory saver (this script doesn't need gradients computation)
torch.set_grad_enabled(False)

######################
# Configs definition #
######################


@dataclass
class TokenizerComparisonConfig:
    # LLMs
    BASE_MODEL_TOKENIZER: str
    FINETUNE_MODEL_TOKENIZER: str

@dataclass
class ActivationStoringConfig:
    # LLMs
    BASE_MODEL: str
    FINETUNE_MODEL: str

    # dataset
    DATASET_NAME: str
    N_CONTEXT: int
    N_BATCHES: int
    TOTAL_BATCHES: int

    # SAE configs
    LAYER_NUM : int 
    SAE_HOOK : str

    # misc
    DTYPE: torch.dtype = torch.float16
    IS_DATASET_TOKENIZED: bool = False

##############################################
# Helper functions to moved in utils or smth #
##############################################

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

### utils function ###

def loading_model_tokenizer_gpu_only_4bit_local(path):
    if not torch.cuda.is_available():
        raise RuntimeError("cuda is off :(")
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print(f"Loading model from: {path}")

    model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map="auto",
                quantization_config=quant_config        
                )
    model.config.use_cache = True
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')
    
    return model, tokenizer


def get_activations_and_tokens(model, LAYER_NUM, SAE_HOOK, TOTAL_BATCHES, DATAPATH, SAVING_NAME_MODEL, SAVING_NAME_DS, N_BATCHES,
                               tokens_loading_path=None, activation_store=None, save=True, tokens_already_loaded=False):
    """
    Get activations and tokens (of which we took the activations) through the model (base or finetuned one)
    """
    if not tokens_already_loaded:
        assert activation_store is not None, "The activation store must be passed for sampling when tokens_already_loaded is False"

        try:
            # If the tokens and activations are already computed, return them
            all_tokens = torch.load(DATAPATH / f"tokens_{SAVING_NAME_MODEL}_on_{SAVING_NAME_DS}.pt")
            all_acts = torch.load(DATAPATH / f"base_acts_{SAVING_NAME_MODEL}_on_{SAVING_NAME_DS}.pt")
            return all_acts, all_tokens
        except:
            # Otherwise compute everything from scratch
            all_tokens = []  # This will store the tokens for reuse
            all_acts = []

            for k in tqdm(range(TOTAL_BATCHES)):
                # Get a batch of tokens from the dataset
                tokens = activation_store.get_batch_tokens()  # [N_BATCH, N_CONTEXT]

                # Store tokens for later reuse
                all_tokens.append(tokens)

                # Run the model and store the activations
                _, cache = model.run_with_cache(tokens, stop_at_layer=LAYER_NUM + 1, \
                                                names_filter=[SAE_HOOK])  # [N_BATCH, N_CONTEXT, D_MODEL]
                all_acts.append(cache[SAE_HOOK])

                # Explicitly free up memory by deleting the cache and emptying the CUDA cache
                del cache
                clear_cache()

            # Concatenate all feature activations into a single tensor
            all_acts = torch.cat(all_acts)  # [TOTAL_BATCHES * N_BATCH, N_CONTEXT, D_MODEL]

            # Concatenate all tokens into a single tensor for reuse
            all_tokens = torch.cat(all_tokens)  # [TOTAL_BATCHES * N_BATCH, N_CONTEXT]

            torch.save(all_tokens, DATAPATH / f"tokens_{SAVING_NAME_MODEL}_on_{SAVING_NAME_DS}.pt")

            if save:
                torch.save(all_acts, DATAPATH / f"base_acts_{SAVING_NAME_MODEL}_on_{SAVING_NAME_DS}.pt")

            return all_acts, all_tokens
        
    # Otherwise, we're dealing with the finetune model and want to load the same tokens sample
    assert tokens_loading_path is not None, "You must provide a path to the sample of tokens for the finetune model when calling this method with tokens_already_loaded=True"

    try:
        all_tokens = torch.load(tokens_loading_path)
    except:
        raise ValueError('A sample of tokens for the finetune model must be already saved at the `all_tokens` path when calling this method with tokens_already_loaded=True')
    
    try:
        all_acts = torch.load(DATAPATH / f"finetune_acts_{SAVING_NAME_MODEL}_on_{SAVING_NAME_DS}.pt")
        return all_acts, all_tokens
    except:
        all_acts = []
        # Split the tokens back into batches and run the fine-tuned model
        for k in tqdm(range(TOTAL_BATCHES)):
            # Calculate the start and end indices for the current batch
            start_idx = k * N_BATCHES
            end_idx = (k + 1) * N_BATCHES

            # Get the corresponding batch of tokens from all_tokens
            tokens = all_tokens[start_idx:end_idx]  # [N_BATCH, N_CONTEXT]

            # Run the fine-tuned model and store the activations
            _, cache = model.run_with_cache(tokens, stop_at_layer=LAYER_NUM + 1, \
                                                    names_filter=[SAE_HOOK])  # [N_BATCH, N_CONTEXT, D_MODEL]
            all_acts.append(cache[SAE_HOOK])

            # Explicitly free up memory by deleting the cache and emptying the CUDA cache
            del cache
            clear_cache()

        # Concatenate all activations from the fine-tuned model into a single tensor
        all_acts = torch.cat(all_acts)  # [TOTAL_BATCHES * N_BATCH, N_CONTEXT, D_MODEL]

        if save:
            torch.save(all_acts, DATAPATH / f"finetune_acts_{SAVING_NAME_MODEL}_on_{SAVING_NAME_DS}.pt")

        return all_acts, all_tokens
    

def get_activations_for_base_and_ft(cfg: ActivationStoringConfig):
    # STEP 1: Get the device and the python and datapath
    device = get_device()
    _, datapath = get_env_var()

    saving_name_base = cfg.BASE_MODEL if "/" not in cfg.BASE_MODEL else cfg.BASE_MODEL.split("/")[-1]
    saving_name_ft = cfg.FINETUNE_MODEL if "/" not in cfg.FINETUNE_MODEL else cfg.FINETUNE_MODEL.split("/")[-1]
    saving_name_ds = cfg.DATASET_NAME if "/" not in cfg.DATASET_NAME else cfg.DATASET_NAME.split("/")[-1]

    # STEP 2: Init the HookedSAETransformer
    base_model = HookedSAETransformer.from_pretrained(cfg.BASE_MODEL, device=device, dtype=cfg.DTYPE)

    # STEP 3: load the config for the activation store
    activation_store_cfg = LanguageModelSAERunnerConfig(
            # Data Generating Function (Model + Training Distibuion)
            model_name=cfg.BASE_MODEL,
            dataset_path=cfg.DATASET_NAME,
            is_dataset_tokenized=cfg.IS_DATASET_TOKENIZED,
            streaming=True,
            # Activation Store Parameters
            store_batch_size_prompts=cfg.N_BATCHES,
            context_size=cfg.N_CONTEXT,
            # Misc
            device=device,
            seed=42,
        )
        
    # STEP 4: Instantiate an activation store to easily sample tokenized batches from our dataset
    activation_store = ActivationsStore.from_config(
        model=base_model,
        cfg=activation_store_cfg
    )
    # model, LAYER_NUM, SAE_HOOK, TOTAL_BATCHES, DATAPATH, SAVING_NAME_MODEL, SAVING_NAME_DS, N_BATCHES

    # STEP 5: Get all activations and tokens through base model
    all_acts, all_tokens = get_activations_and_tokens(base_model, cfg.LAYER_NUM, cfg.SAE_HOOK, cfg.TOTAL_BATCHES, datapath,
                                                      saving_name_base, saving_name_ds, cfg.N_BATCHES, activation_store=activation_store)

    # STEP 6: Offload the first model from memory, but save its tokenizer
    base_tokenizer = base_model.tokenizer
    del base_model, activation_store # also delete activation store as it has base_model captured as a parameter
    clear_cache()

    # STEP 7: Load the finetuned model
    finetune_tokenizer = AutoTokenizer.from_pretrained(cfg.FINETUNE_MODEL)
    finetune_model_hf = AutoModelForCausalLM.from_pretrained(cfg.FINETUNE_MODEL)
    finetune_model = HookedSAETransformer.from_pretrained(cfg.BASE_MODEL, device=device, hf_model=finetune_model_hf, dtype=cfg.DTYPE)

    del finetune_model_hf # offload the finetune HF models because it's already wrapped into HookedSAETransformer (finetune_model)
    clear_cache()

    # STEP 8: Get all activations through finetuned model
    # We should use the same sample of tokens as in the first get_activations_and_tokens() call
    tokens_loading_path = datapath / f"tokens_{saving_name_base}_on_{saving_name_ds}.pt"

    all_acts_finetuned, all_tokens = get_activations_and_tokens(finetune_model, cfg.LAYER_NUM, cfg.SAE_HOOK, cfg.TOTAL_BATCHES, datapath,
                                                                saving_name_ft, saving_name_ds, cfg.N_BATCHES, tokens_already_loaded=True,
                                                                tokens_loading_path=tokens_loading_path)

    return {
        "base_act_path" : datapath / f"base_acts_{saving_name_base}_on_{saving_name_ds}.pt",
        "finetune_act_path" : datapath / f"finetune_acts_{saving_name_ft}_on_{saving_name_ds}.pt",
        "base_tokenizer" : base_tokenizer, 
        "finetune_tokenizer" : finetune_tokenizer
    }


def compare_tokenizers(cfg: TokenizerComparisonConfig):
    base_model_tok_name = cfg.BASE_MODEL_TOKENIZER
    finetune_model_tok_name = cfg.FINETUNE_MODEL_TOKENIZER
    saving_name_ft = cfg.FINETUNE_MODEL_TOKENIZER if "/" not in cfg.FINETUNE_MODEL_TOKENIZER else cfg.FINETUNE_MODEL_TOKENIZER.split("/")[-1]
    
    _, datapath = get_env_var()
    saving_path = datapath / "log" /f'{saving_name_ft}_tokenizer_vocab_comparison_log.txt'
    if not os.path.exists(datapath / "log"):
        os.makedirs(datapath / "log")

    base_tokenizer = AutoTokenizer.from_pretrained(base_model_tok_name)
    finetune_tokenizer = AutoTokenizer.from_pretrained(finetune_model_tok_name)

    # Setup the file-logger
    logger = logging.getLogger('tokenizer_vocab_comparison')

    # Clear any existing handlers to ensure no console logging
    logger.handlers.clear()

    # Set the log level
    logger.setLevel(logging.INFO)

    # Create file handler with UTF-8 encoding
    file_handler = logging.FileHandler(saving_path, encoding='utf-8')

    # Set the logging format
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add only the file handler to the logger
    logger.addHandler(file_handler)

    # Disable propagation to prevent any parent loggers from printing to the console
    logger.propagate = False

    # Extract vocabs
    base_vocab = base_tokenizer.get_vocab()
    finetune_vocab = finetune_tokenizer.get_vocab()

    # Run the vocab comparison code
    # 1. Compare the keys (words/tokens)
    base_keys = set(base_vocab.keys())
    finetune_keys = set(finetune_vocab.keys())
    
    # Keys that are in one tokenizer but not in the other
    only_in_base = base_keys - finetune_keys
    only_in_finetune = finetune_keys - base_keys
    
    logger.info("Keys only in base tokenizer:")
    for key in only_in_base:
        logger.info(f"  {key}")
    
    logger.info("\nKeys only in fine-tuned tokenizer:")
    for key in only_in_finetune:
        logger.info(f"  {key}")
    
    # 2. Compare the values (token ids)
    mismatched_values = {}
    for key in base_keys.intersection(finetune_keys):
        base_value = base_vocab[key]
        finetune_value = finetune_vocab[key]
        if base_value != finetune_value:
            mismatched_values[key] = (base_value, finetune_value)
    
    logger.info("\nKeys with mismatched token IDs:")
    for key, (base_value, finetune_value) in mismatched_values.items():
        logger.info(f"  {key}: Base ID = {base_value}, Fine-tune ID = {finetune_value}")

    # Ensure the log is flushed
    for handler in logger.handlers:
        handler.flush()

    # Define variables based on results
    base_vocab_size = len(base_vocab)
    finetune_vocab_size = len(finetune_vocab)
    only_in_base_size = len(only_in_base)
    only_in_finetune_size = len(only_in_finetune)
    mismatched_values_size = len(mismatched_values)

    # Calculate good token counts
    good_base_tokens_count = base_vocab_size - only_in_base_size - mismatched_values_size
    good_finetune_tokens_count = finetune_vocab_size - only_in_finetune_size - mismatched_values_size

    # Calculate percentages
    good_base_tokens_percent = good_base_tokens_count / base_vocab_size * 100
    good_finetune_tokens_percent = good_finetune_tokens_count / finetune_vocab_size * 100

    # Summary statistics
    summary_statistics = {
        "Base Tokenizer Size": base_vocab_size,
        "Fine-tune Tokenizer Size": finetune_vocab_size,
        "Keys only in Base": only_in_base_size,
        "Keys only in Fine-tune": only_in_finetune_size,
        "Keys with Mismatched Token IDs": mismatched_values_size,
        "Good Tokens in Base (%)": good_base_tokens_percent,
        "Good Tokens in Fine-tune (%)": good_finetune_tokens_percent
    }

    # Create a pandas DataFrame for display
    summary_df = pd.DataFrame(list(summary_statistics.items()), columns=["Metric", "Value"])
    logger.info(str(summary_df))

    return summary_df

if __name__ == "__main__":
    ### set this args with argparse, now hardcoded
    GEMMA=False

    if GEMMA == True:
        N_CONTEXT = 1024 # number of context tokens to consider
        N_BATCHES = 128 # number of batches to consider
        TOTAL_BATCHES = 20 

        RELEASE = 'gemma-2b-res-jb'
        BASE_MODEL = "google/gemma-2b"
        FINETUNE_MODEL = 'shahdishank/gemma-2b-it-finetune-python-codes'
        BASE_TOKENIZER_NAME = BASE_MODEL
        DATASET_NAME = "ctigges/openwebtext-gemma-1024-cl"
        hook_part = 'post'
        layer_num = 6

    else:
        N_CONTEXT = 128 # number of context tokens to consider
        N_BATCHES = 128 # number of batches to consider
        TOTAL_BATCHES = 100 

        RELEASE = 'gpt2-small-res-jb'
        BASE_MODEL = "gpt2-small"
        FINETUNE_MODEL = 'pierreguillou/gpt2-small-portuguese'
        BASE_TOKENIZER_NAME = 'openai-community/gpt2'
        DATASET_NAME = "Skylion007/openwebtext"
        hook_part = 'pre'
        layer_num = 6

    SAE_HOOK = f'blocks.{layer_num}.hook_resid_{hook_part}'

    tokenizer_cfg = TokenizerComparisonConfig(BASE_TOKENIZER_NAME, FINETUNE_MODEL)

    tokenizer_comp_df = compare_tokenizers(tokenizer_cfg)
    print(tokenizer_comp_df)

    activation_cfg = ActivationStoringConfig(BASE_MODEL, FINETUNE_MODEL, DATASET_NAME, 
                                             N_CONTEXT, N_BATCHES, TOTAL_BATCHES, 
                                             layer_num, SAE_HOOK)

    result_dict = get_activations_for_base_and_ft(activation_cfg)
    print(result_dict)

    base_act = torch.load(result_dict['base_act_path'])
    finetune_act = torch.load(result_dict['finetune_act_path'])

    print('Base vs finetune activations shape: ')
    print(base_act.shape, finetune_act.shape)