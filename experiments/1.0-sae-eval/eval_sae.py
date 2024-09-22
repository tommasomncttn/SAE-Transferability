import argparse
from pathlib import Path
import yaml
import sys 
import os 
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
    cfg = ScoresConfig(**cfg)

    # comptue the scores
    compute_scores(cfg)