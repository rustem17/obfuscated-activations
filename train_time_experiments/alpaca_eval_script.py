import os
import sys
import torch
from pathlib import Path

# Third‑party imports
import numpy as np
from datasets import load_dataset
from huggingface_hub import snapshot_download
from peft import PeftModel

# Import evaluation utilities
from transformers import AutoTokenizer, GenerationConfig
from alpaca_eval import evaluate as alpaca_evaluate
from alpaca_eval import analyze

# --- Local imports (modify these paths as needed) ---
# These modules come from your repository in src/
from src.probe_evals import *
from src.probe_archs import *
from src.probe_training import *
# (Any other local helper functions can be imported too)

# Check that the OpenAI API key is set in the environment.
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_KEY is None:
    sys.exit("Error: OPENAI_API_KEY is not set in the environment.")

# Device configuration (you may adjust the visible devices as needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Download/Locate Repository
# ---------------------------
# This repo holds both the LoRA adapter and the custom probes.
repo_id = "Mechanistic-Anomaly-Detection/llama3-oat-generation-linear"
local_dir = "llama3-oat-generation-linear"
snapshot_download(repo_id=repo_id, local_dir=local_dir)

# ---------------------------
# Model Loading Functions
# ---------------------------
# In this example we assume that your models come from an EleutherSparseAutoencoder helper,
# which loads a LLaMa‑3 model and its tokenizer.

# Import your encoder class (adjust as needed)
from src import EleutherSparseAutoencoder

def load_baseline():
    """
    Load the baseline LLaMa‑3 model.
    We assume the baseline is the adapter‑merged (unmodified) model.
    """
    # Load the model with instructions enabled.
    encoder = EleutherSparseAutoencoder.load_llama3_sae(None, instruct=True)

    # If the model still has a LoRA adapter (PeftModel), merge and unload it.
    if isinstance(encoder.model, PeftModel):
        encoder.model = encoder.model.merge_and_unload()

    # Move model to proper device
    encoder.model.to(DEVICE)
    return encoder.model, encoder.tokenizer

def load_modified():
    """
    Load the modified model: start from the same LLaMa‑3 model and then attach
    the LoRA adapter (if needed) and your custom probes.
    """
    # Load base model first.
    encoder = EleutherSparseAutoencoder.load_llama3_sae(None, instruct=True)
    encoder.model.to(DEVICE)

    # If a LoRA adapter is not already applied, load it.
    if not isinstance(encoder.model, PeftModel):
        encoder.model = PeftModel.from_pretrained(encoder.model, local_dir)
        # (Optionally merge later if required for inference.)

    # Load probes (your implementation must match the architecture).
    # Assume load_probes returns a nn.Module, for example a linear probe.
    probes_path = os.path.join(local_dir, "probes.pt")
    linear_probes = load_probes(probes_path)

    # Attach the probes to the model (here we simply add an attribute).
    encoder.model.probe = linear_probes.to(DEVICE)

    return encoder.model, encoder.tokenizer

# ---------------------------
# Generate Responses for Evaluation
# ---------------------------
def generate_responses(model, tokenizer, output_path):
    """
    Generate responses using the alpaca_eval helper.
    (Make sure your model’s generate() method follows the transformers interface.)
    """
    return alpaca_evaluate(
        model=model,
        tokenizer=tokenizer,
        dataset="alpaca_eval",  # Name of your evaluation dataset
        output_path=output_path,
        generation_config=GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=256
        ),
        batch_size=4  # Adjust as needed to prevent OOM issues
    )

# ---------------------------
# Main Procedure: Run Evaluation & Example Inference
# ---------------------------
if __name__ == "__main__":
    # --- Evaluate models ---
    print("Loading baseline model …")
    baseline_model, tokenizer = load_baseline()

    print("Loading modified model …")
    modified_model, _ = load_modified()  # using the same tokenizer

    print("Generating evaluation outputs …")
    _ = generate_responses(baseline_model, tokenizer, "baseline.json")
    _ = generate_responses(modified_model, tokenizer, "modified.json")

    print("Running analysis …")
    results = analyze(
        outputs_1="modified.json",
        outputs_2="baseline.json",
        annotators_config="alpaca_eval_gpt4",
        openai_api_key=OPENAI_KEY,
        max_instances=805  # Use the full dataset or adjust as needed
    )
    print("Results:\n", results)
