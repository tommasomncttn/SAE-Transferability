import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
import os

PROJECT_ROOT = "/leonardo_work/IscrC_MGNTC/tmencatt/SAE-Tuning-Merging"
CACHE_DIR = f"{PROJECT_ROOT}/cache_dir"

def save_model_local(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, name_model: str) -> None:
    
    # build the path to save the model
    root = Path(f"{PROJECT_ROOT}")
    saving_path = root / "models" / name_model

    # Check if the path exists before saving the model
    for _, dirs, files in os.walk(saving_path): #_ is root
        if files or dirs:
            raise RuntimeError("empty the directory before, is not empty")

    # Save the model and tokenizer
    print(f"saving the model here: {saving_path}")
    model.save_pretrained(saving_path)
    tokenizer.save_pretrained(saving_path)

def loading_model_tokenizer(name_model, cache_dir = f"{PROJECT_ROOT}/cache_dir"):

    if not torch.cuda.is_available():
        raise RuntimeError("cuda is off :(")
    
    cache_dir = str(Path(cache_dir) / name_model)

    model = AutoModelForCausalLM.from_pretrained(
                name_model,
                cache_dir = cache_dir,
                torch_dtype = torch.float32             
                )
    model.config.use_cache = True
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(name_model, padding_side='left')
    
    return model, tokenizer

def main(hugging_model: str):

    # Loading the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
                hugging_model,
                cache_dir = f"{CACHE_DIR}/{hugging_model}",
                torch_dtype = torch.float16              
                )
    tokenizer = AutoTokenizer.from_pretrained(hugging_model, padding_side='left')

    # not sure this is needed
    model.config.use_cache = True
    model.eval()

    # Adjusting generation config
    if model.generation_config.do_sample is False and model.generation_config.temperature != 1.0:
        model.generation_config.do_sample = True  # or set temperature to 1.0

    save_model_local(model, tokenizer, name_model=hugging_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and save a Hugging Face model and tokenizer.")
    parser.add_argument('hugging_model', type=str, help='The name of the Hugging Face model to load')
    
    args = parser.parse_args()
    main(args.hugging_model)