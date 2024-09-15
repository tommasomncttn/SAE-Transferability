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

##############################################
# Helper functions to moved in utils or smth #
##############################################


### set this args with argparse, now hardcoded
GEMMA=False

if GEMMA == True:
    N_CONTEXT = 1024 # number of context tokens to consider
    N_BATCHES = 128 # number of batches to consider
    TOTAL_BATCHES = 20 

    RELEASE = 'gemma-2b-res-jb'
    BASE_MODEL = "google/gemma-2b"
    FINETUNE_MODEL = 'shahdishank/gemma-2b-it-finetune-python-codes'
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
    DATASET_NAME = "Skylion007/openwebtext"
    hook_part = 'pre'
    layer_num = 6

@dataclass
class TokenizerComparisonConfig:
    # LLMs
    BASE_MODEL: str
    FINETUNE_MODEL: str

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


def parse_batches(activation_store, model, LAYER_NUM, SAE_HOOK, TOTAL_BATCHES, datapath, SAVING_NAME_MODEL, SAVING_NAME_DS, save = True, tokens_already_loaded = False):
    """
    get activations and tokens (of which we took the activation) through the base model
    """
    all_acts = []

    if not tokens_already_loaded:

        try:
            all_tokens = torch.load(datapath + f"/tokens_{SAVING_NAME_MODEL}_on_{SAVING_NAME_DS}.pt")
            all_acts = torch.load(datapath + f"/base_acts_{SAVING_NAME_MODEL}_on_{SAVING_NAME_DS}.pt")
            return all_acts, all_tokens
        except:
            # check if file already exists and load it
            all_tokens = []  # This will store the tokens for reuse

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
                torch.cuda.empty_cache()

            # Concatenate all feature activations into a single tensor
            all_acts = torch.cat(all_acts)  # [TOTAL_BATCHES * N_BATCH, N_CONTEXT, D_MODEL]

            # Concatenate all tokens into a single tensor for reuse
            all_tokens = torch.cat(all_tokens)  # [TOTAL_BATCHES * N_BATCH, N_CONTEXT]

            if save:
                torch.save(all_tokens, datapath + f"/tokens_{SAVING_NAME_MODEL}_on_{SAVING_NAME_DS}.pt")
                torch.save(all_acts, datapath + f"/base_acts_{SAVING_NAME_MODEL}_on_{SAVING_NAME_DS}.pt")

            return all_acts, all_tokens
    
    else:
        try:
            all_acts = torch.load(datapath + f"/finetune_acts_{SAVING_NAME_MODEL}_on_{SAVING_NAME_DS}.pt")
            return all_acts
        except:
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
                torch.cuda.empty_cache()

            # Concatenate all activations from the fine-tuned model into a single tensor
            all_acts = torch.cat(all_acts)  # [TOTAL_BATCHES * N_BATCH, N_CONTEXT, D_MODEL]

            if save:
                torch.save(all_acts, datapath + f"/finetune_acts_{SAVING_NAME_MODEL}_on_{SAVING_NAME_DS}.pt")

            return all_acts
    

def get_activations_for_base_and_ft(cfg: ActivationStoringConfig):

    # STEP 1: Get the device and the python and datapath
    device = get_device()
    pythonpath, datapath = get_env_var()
    saving_name_base = cfg.BASE_MODEL if "/" not in cfg.BASE_MODEL else cfg.BASE_MODEL.split("/")[-1]
    saving_name_ft = cfg.FINETUNE_MODEL if "/" not in cfg.FINETUNE_MODEL else cfg.FINETUNE_MODEL.split("/")[-1]
    saving_name_ds = cfg.DATASET_NAME if "/" not in cfg.DATASET_NAME else cfg.DATASET_NAME.split("/")[-1]


    # STEP 2: Load the dataset
    try:
        dataset = load_dataset(cfg.DATASET_NAME, split="train", streaming=True)
    except:
        dataset = load_dataset(cfg.DATASET_NAME, streaming=True) 
    
    # STEP 3: Init the HookedSAETransformer
    base_model = HookedSAETransformer.from_pretrained(cfg.BASE_MODEL, device=device, dtype=cfg.DTYPE)

    # STEP 4: load the config for the activation store
    cfg = LanguageModelSAERunnerConfig(
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
        
    # STEP 5: Instantiate an activation store to easily sample tokenized batches from our dataset
    activation_store = ActivationsStore.from_config(
        model=base_model,
        cfg=cfg
    )

    # STEP 6: Get all activations and tokens through base model
    all_acts, all_tokens = parse_batches(activation_store, base_model, cfg.LAYER_NUM, cfg.SAE_HOOK, cfg.TOTAL_BATCHES, datapath, saving_name_base, saving_name_ds)

    # STEP 7: Offload the first model from memory, but save its tokenizer
    base_tokenizer = base_model.tokenizer
    del base_model
    torch.cuda.empty_cache()

    # STEP 8: Load the finetuned model
    finetune_tokenizer = AutoTokenizer.from_pretrained(cfg.FINETUNE_MODEL)
    finetune_model_hf = AutoModelForCausalLM.from_pretrained(FINETUNE_MODEL)
    finetune_model = HookedSAETransformer.from_pretrained(cfg.BASE_MODEL, device=device, hf_model=finetune_model_hf, dtype=cfg.DTYPE)

    # STEP 9: Get all activations through finetuned model
    all_acts_finetuned = parse_batches(activation_store, finetune_model, cfg.LAYER_NUM, cfg.SAE_HOOK, cfg.TOTAL_BATCHES, datapath, saving_name_ft, saving_name_ds, tokens_already_loaded = True)

    return {"base_act_path" : datapath + f"/base_acts_{saving_name_base}_on_{saving_name_ds}.pt", "finetune_act_path" : datapath + f"/finetune_acts_{saving_name_ft}_on_{saving_name_ds}.pt", "base_tokenizer" : base_tokenizer, "finetune_tokenizer" : finetune_tokenizer}


def compare_tokenizers(cfg: TokenizerComparisonConfig):
    base_model_name = cfg.BASE_MODEL
    finetune_model_name = cfg.FINETUNE_MODEL
    saving_name_ft = cfg.FINETUNE_MODEL if "/" not in cfg.FINETUNE_MODEL else cfg.FINETUNE_MODEL.split("/")[-1]
    
    pythonpath, datapath = get_env_var()
    saving_path = datapath / "log" /f'{saving_name_ft}_tokenizer_vocab_comparison_log.txt'

    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    finetune_tokenizer = AutoTokenizer.from_pretrained(finetune_model_name)

    # Setup the file-logger
    logger = logging.getLogger('tokenizer_vocab_comparison')

    # Clear any existing handlers to ensure no console logging
    logger.handlers.clear()

    # Set the log level
    logger.setLevel(logging.INFO)

    # Create file handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8')

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

    logger.info('\nPercentage of good tokens in the base vocab: ', good_base_tokens_percent)
    logger.info('\nPercentage of good tokens in the finetune vocab: ', good_finetune_tokens_percent)

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