#!/bin/bash
#SBATCH -A IscrC_MGNTC
#SBATCH --time 24:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gres=gpu:1        # 4 gpus per node out of 4
#SBATCH --mem=48000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=translate_gsm8k_dutch
#SBATCH --error=/leonardo_work/IscrC_MGNTC/tmencatt/SAE-Tuning-Merging/experiments/1.0-sae-eval/sae_eval.err   # standard error file
#SBATCH --output=/leonardo_work/IscrC_MGNTC/tmencatt/SAE-Tuning-Merging/experiments/1.0-sae-eval/sae_eval.out   # standard output file
#SBATCH --partition=boost_usr_prod

# load modules, activate environment, and run the script
# ...
source activate saemerging
cd /leonardo_work/IscrC_MGNTC/tmencatt/SAE-Tuning-Merging/experiments/1.0-sae-eval
python eval_sae.py --config_yaml "/leonardo_work/IscrC_MGNTC/tmencatt/SAE-Tuning-Merging/experiments/1.0-sae-eval/score_config_gemma.yaml"

