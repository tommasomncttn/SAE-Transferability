# What is my purpose

This is a code repo for getting the results for our [SAEs transferability post](https://www.lesswrong.com/posts/bsXPTiAhhwt5nwBW3/do-sparse-autoencoders-saes-transfer-across-base-and) - check the TLDR section for a short summary!

# How to use me

All the code that was used for getting the results of the aforementioned post is **located in the `notebooks/` folder**. We experimented with running our experiments on a remote cluster, but ultimately decided to stick with notebooks due to some technical difficulties.

## Prerequisites

1. The notebooks are intended to be used in **Google Colab**. For Gemma-2b and Mistral-7B you will need a standard paid subscription to run these models (T4 GPU is recommended for Gemma; L4 GPU for Mistral). They *could* be modified to be used locally (and it's likely that we will do it in the future), more on that below. 
	- For prototyping though, it's possible to run them on the free Colab version for GPT2-small. 
2. All notebooks use **Google Drive** to save & load data for different experiments, and optionally for loading finetuned models. 
	- Every notebook **mounts** your google drive (by default, from the same account that you're using Colab from) and saves the data to your `My Drive/sae_data` folder by default, so **make sure to create this folder before running the notebooks.**
	- Of course it's possible to **use another drive folder** (or another local folder if you've modified the notebooks accordingly): the place to search for is **the `datapath` variable** (In Colab it's best to search by pressing the 'lens' button on the left panel).
## Instruction

After making sure that you've met the above prerequisites, here you can find a step-by-step on how to reproduce/extend our results from each section of our [post](modified).

In all our notebooks, you'll need to fill in the following (hard-coded) config in the **Config** section of each notebook:

1. **`BASE_MODEL`**:  
   Specifies the base model to be used (e.g., `"google/gemma-2b"`). **This is an input to the HookedSAETransformer** class, so the **`BASE_MODEL`** should be one of the [supported models from SAELens](https://jbloomaus.github.io/SAELens/sae_table/)
2. **`DATASET_NAME`**:  
   Defines the dataset of the corresponding SAE (e.g. `"ctigges/openwebtext-gemma-1024-cl"`). It's used only for logging purposes, because **all the notebooks use the training dataset of the corresponding SAE by default**.
	   - If you want to use other datasets, you can load (or create) your custom HF dataset and pass it as a parameter to the `activation_store` instance ([SAELens object](https://jbloomaus.github.io/SAELens/api/#sae_lens.ActivationsStore) for sampling tokenized batches from a HF dataset), like this:
   ```
		activation_store = ActivationsStore.from_sae(
		    model=model,
		    sae=sae,
		    dataset=your_hf_dataset,
		    ...
   ```
4. **`BASE_TOKENIZER_NAME`**:  
   Refers to the tokenizer name of the `BASE_MODEL`. Should be a valid HF repo name that contains tokenizer files (typically the same repo as the base language model, e.g. `google/gemma-2b`). It is used only in *pre_3_compute_activations.ipynb* notebook to compare tokenizers of the base and finetuned models.
5. **`FINETUNE_MODEL`**:  
   Identifier of the finetuned model (e.g., `'shahdishank/gemma-2b-it-finetune-python-codes'`). This should be a valid HuggingFace model name from the [Model hub](https://huggingface.co/models), and the corresponding model should be a finetune of the `BASE_MODEL`.
6. **`FINETUNE_PATH`**:  
   If you execute the *save_finetune_model.ipynb* notebook and save the model to your Google Drive, specify the path to it here. Leave it None if you want to download the finetune from the HF directly.
7. **`RELEASE`**:  
   Name of the SAE release for the `BASE_MODEL`. Must be [one of the supported SAELens releases](https://jbloomaus.github.io/SAELens/sae_table/).
   
   The two remaining parameters are only meant to be used for SAEs trained on **residual streams**. They are used for `sae_id` variable definition to load a residual SAE. **Please modify your `sae_id` variable in the notebooks *4_sae_eval.ipynb* and *5_features_transfer.ipynb* if your want to use SAEs for other kinds of model activations.**
   
8. **`hook_part`**:  
   Specifies the part of the residual stream for hooking: `'post'` or `'pre'`, depending on your `sae_id` format.
9. **`layer_num`**:  
   Specifies the residual layer number of the corresponding SAE.
   
Currently, the config cells are prefilled to work with 3 models (Mistral-7B, Gemma-2b, GPT2-small), but you can edit them with whatever valid values you like.

### Computing activations similarity 
#### Computing the activations
1. Make sure you read the Prerequisites section and set up your Colab & Google Drive.
2. Open the *pre_3_compute_activations.ipynb* notebook and fill in the config cell (easily found by the header name in the table of contents).
3. Run the notebook *pre_3_compute_activations.ipynb*.

**Important**: the notebook expects the base and finetuned model to have identical tokenizer vocabularies. If this is not the case, you will see something like this as a result of the *Tokenizers test* section execution:
| Metric                         | Value      |
|--------------------------------|------------|
| Base Tokenizer Size            | 50257.00000 |
| Fine-tune Tokenizer Size       | 50257.00000 |
| Keys only in Base              | 37309.00000 |
| Keys only in Fine-tune         | 37309.00000 |
| Keys with Mismatched Token IDs | 12946.00000 |
| Good Tokens in Base (%)        | 0.00398     |
| Good Tokens in Fine-tune (%)   | 0.00398     |

By "Keys" here we mean text strings (that get mapped to their *values*: integer token ids), and the `Good tokens in model X` is defined as `X_vocab_size - tokens_only_in_X - same_tokens_with_different_ids'.

**Note that the notebook will still continue to run** because it uses the same tokenizer from the base model for **both base and finetuned** models. The way it works technically is that the input text is tokenized by the base tokenizer to get the input tokens ids, and **the same token ids** **are passed to the finetuned model**. This doesn't make any sense if the finetuned model has different vocabulary!

P.S. sometimes the finetuned model is extended with just a few new tokens in the vocabulary: this is the case for our Mistral-7B MetaMath finetune which introduced a single new PAD token. This shouldn't be a problem if you're running both models on the base SAE dataset as it is done in the blog post. However, one should discard the new tokens in the finetuned model to be able to load it into *HookedSAETransformer* due to how the class is implemented. This is done in the `load_hf_model` method across all notebooks.
#### Computing the similarities
1. Run the *pre_3_compute_activations.ipynb* as described in the previous step.
2. Copy the config from *pre_3_compute_activations.ipynb* into the corresponding Config section (found in the table of contents) of *3_compute_similarities.ipynb*.
3. Run the *3_compute_similarities.ipynb*.

## Evaluating SAEs performance on the finetuned model
1. Run the *pre_3_compute_activations.ipynb* to check if the tokenizers of the base and finetuned models are identical (for the reason described above).
	- **TODO:** this is a major limitation which doesn't make much sense for our *4_sae_eval.ipynb* and *5_features_transfer.ipynb* notebooks (unlike for computing activation similarities). It's worth considering modifying these notebooks to use the finetuned model tokenizers.
2. [As outlined in our post](https://lesswrong.com/posts/bsXPTiAhhwt5nwBW3/do-sparse-autoencoders-saes-transfer-across-base-and#4_2_Technical_Details), for this task we need to filter out outlier activations to avoid numerical instability. Our filtering implementation relies on loading the `outlier_cfg.json` file from Google drive (from the base 'My Drive' folder by default). 
	- You can download this file from our Github repo (`outlier_cfg.json` in the root folder) for the models we used in the post, and upload it to your Google Drive
	- If you want to run the notebook for your own models, you should either provide an **absolute threshold** (above which the activations will be excluded from analysis), or run the *find_outlier_norms.ipynb* notebook before to use **relative threshold**. See the difference in the doc string of the `is_act_outlier()` method of *4_sae_eval.ipynb* notebook.
3. Open the *4_sae_eval.ipynb* and fill in its config cell.
4. Run the *4_sae_eval.ipynb* notebook.

## Testing the SAE features transferability
1. Run the *4_sae_eval.ipynb* notebook to compute feature densities.
2. Open the *5_features_transfer.ipynb* notebook and fill in its config cell.
3. Specify how many features to sample in the *Sampling features from density intervals* section.
4. Run the *5_features_transfer.ipynb* notebook.

Thus, our notebooks have the following dependencies in terms of data input/outputs:
- *pre_3_compute_activations.ipynb* -> *3_compute_similarities.ipynb*
- (check the output of *pre_3_compute_activations.ipynb*) -> *4_sae_eval.ipynb* -> *5_features_transfer.ipynb*

## Controlling sample sizes
We use [ActivationStore](https://jbloomaus.github.io/SAELens/api/#sae_lens.ActivationsStore) class instances for tokens sampling, so all the sample sizes for our experiments are controlled by the parameters of the corresponding `activation_store` variable:
- `store_batch_size_prompts`: how many text sequences to run through the model at once.
- `context_size`: the length of the text sequences to run through the model. Defaults to the `context_size` that the corresponding SAE was trained on (revealed by its config that is returned by SAE.from_pretrained() method from SAELens)
- `total_batches` (and its variations): all our experiment have this parameter (defined either as a function parameter or as a global variable) to control how my 'outer batches' to run through the model sequentially. We'd like to run all the 256K tokens through the model at once, but unluckily our VRAM is not infinite! That's why we adjusted this parameter for each experiment to control **how many forward passes to perform with our `[store_batch_size_prompts, context_size]`-shaped inputs.**

# Other code
There's a bunch of other scripts/notebooks in the repo, but they are poorly documented and were used mostly for prototyping or when we tried to run our experiments on the remote cluster (*saetuning* folder). You can check them if you want, but we have warned you! :D
