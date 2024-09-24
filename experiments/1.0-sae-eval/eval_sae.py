import argparse
import yaml
import sys 
import os 
import torch
from datasets import load_from_disk
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '/leonardo_work/IscrC_MGNTC/tmencatt/SAE-Tuning-Merging/saetuning')))

if __name__ == "__main__":
    from get_scores import ScoresConfig, compute_scores

    ### set this args with argparse, now hardcoded
    parser = argparse.ArgumentParser(description='Run GA optimization with configurable parameters')

    parser.add_argument('--config_yaml', type=str, default=False, help='Path to the yaml configuration file to instatiate Scoreconfig')

    # open the yaml file into a dictionary to create a ScoresConfig object
    args = parser.parse_args()
    if args.config_yaml:
        with open(args.config_yaml, 'r') as f:
            cfg = yaml.safe_load(f)

    # create the Score Config object
    print(cfg)
    config_obj = {
        "BASE_MODEL":cfg.get('BASE_MODEL', None),
        "FINETUNE_MODEL":cfg.get('FINETUNE_MODEL', None),
        "DATASET_NAME":cfg.get('DATASET_NAME', None),
        "SAE_RELEASE":cfg.get('SAE_RELEASE', None),
        "LAYER_NUM":cfg.get('LAYER_NUM', 0),
        "HOOK_PART":cfg.get('HOOK_PART', None),
        "DTYPE":torch.float16 if cfg.get('DTYPE') == 'float16' else torch.float32,
        "IS_DATASET_TOKENIZED":cfg.get('IS_DATASET_TOKENIZED', False),
        "SUBSTITUTION_LOSS_BATCH_SIZE":cfg.get('SUBSTITUTION_LOSS_BATCH_SIZE', 25),
        "L0_LOSS_BATCH_SIZE":cfg.get('L0_LOSS_BATCH_SIZE', 50),
        "FEATURE_ACTS_BATCH_SIZE":cfg.get('FEATURE_ACTS_BATCH_SIZE', 25),
        "FEATURE_DENSITY_BATCH_SIZE":cfg.get('FEATURE_DENSITY_BATCH_SIZE', 50),
        "STORE_BATCH_SIZE_PROMPTS":cfg.get('STORE_BATCH_SIZE_PROMPTS', 8),
        "TRAIN_BATCH_SIZE_TOKENS":cfg.get('TRAIN_BATCH_SIZE_TOKENS', 4096),
        "N_BATCHES_IN_BUFFER":cfg.get('N_BATCHES_IN_BUFFER', 32),
        "N_BATCH_TOKENS":cfg.get('N_BATCH_TOKENS', None),
        "HF_MODEL_BASE":cfg.get('HF_MODEL_BASE', None),
        "HF_MODEL_FINETUNE":cfg.get('HF_MODEL_FINETUNE', None),
        "DATASET_PATH":cfg.get('DATASET_PATH', None)
    }

    # load the dataset as a hf dataset from my path /path/to/dataset
    dataset = load_from_disk(cfg.DATASET_PATH)
    # comptue the scores
    compute_scores(cfg)