import os
import sys
import json
import torch
from pathlib import Path

# Third‑party imports
import numpy as np
from datasets import load_dataset
from huggingface_hub import snapshot_download
from peft import PeftModel

# Import evaluation utilities (if still needed later)
from transformers import AutoTokenizer, GenerationConfig
# You may remove/import analyze_evaluators if you later want to run another analysis step.
from alpaca_eval import analyze_evaluators as analyze

# --- Local imports (modify these paths as needed) ---
from src.probe_evals import *
from src.probe_archs import *
from src.probe_training import *
from src import EleutherSparseAutoencoder  # Import your encoder class

# Check that the OpenAI API key is set in the environment.
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_KEY is None:
    sys.exit("Error: OPENAI_API_KEY is not set in the environment.")

# Device configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Download/Locate Repository
# ---------------------------
repo_id = "Mechanistic-Anomaly-Detection/llama3-oat-generation-linear"
local_dir = "llama3-oat-generation-linear"
snapshot_download(repo_id=repo_id, local_dir=local_dir)

# ---------------------------
# Model Loading Functions
# ---------------------------
def load_baseline():
    """
    Load the baseline LLaMa‑3 model.
    Assumes the baseline has the adapter merged.
    """
    encoder = EleutherSparseAutoencoder.load_llama3_sae(None, instruct=True)
    if isinstance(encoder.model, PeftModel):
        encoder.model = encoder.model.merge_and_unload()
    encoder.model.to(DEVICE)
    encoder.model.eval()  # set eval mode
    return encoder.model, encoder.tokenizer

def load_modified():
    """
    Load the modified model: attach the LoRA adapter and custom probes.
    """
    encoder = EleutherSparseAutoencoder.load_llama3_sae(None, instruct=True)
    encoder.model.to(DEVICE)
    if not isinstance(encoder.model, PeftModel):
        encoder.model = PeftModel.from_pretrained(encoder.model, local_dir)
    probes_path = os.path.join(local_dir, "probes.pt")
    linear_probes = load_probes(probes_path)
    for key, module in linear_probes.items():
        linear_probes[key] = module.to(DEVICE)
    encoder.model.probe = linear_probes
    encoder.model.eval()  # set eval mode
    return encoder.model, encoder.tokenizer

# ---------------------------
# Generate Responses for Evaluation
# ---------------------------
def generate_responses(model, tokenizer, output_path):
    """
    Load the 'alpaca_eval' dataset and generate responses using a custom inference loop.
    For each sample, the input text is processed using the chat template function
    and then passed to the model.generate() call.
    The generated responses are collected and dumped into a single JSON file.
    """
    # Load the dataset (adjust the split if needed)
    dataset = load_dataset("tatsu-lab/alpaca_eval", split="eval")

    responses = []
    # Loop over each sample in the dataset.
    for sample in dataset:
        # Adjust the key names below if your dataset uses different keys.
        # Here we assume each sample has an "instruction" field.
        input_text = sample.get("instruction", "")
        if not input_text:
            continue  # skip if no input text

        # Use the provided inference code snippet:
        tokens = tokenizer.apply_chat_template(
            [{"content": input_text, "role": "user"}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(DEVICE)
        generated_ids = model.generate(tokens, max_new_tokens=100, do_sample=True)
        response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        responses.append({
            "instruction": input_text,
            "response": response_text
        })

    # Save responses to a single JSON file.
    with open(output_path, "w") as f:
        json.dump(responses, f, indent=2)
    return responses

def print_response_summary(json_file, model_label):
    """
    Print a summary of the responses stored in the JSON file.
    """
    try:
        with open(json_file, "r") as f:
            responses = json.load(f)
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return

    num = len(responses) if isinstance(responses, list) else 1
    print(f"{model_label} responses in {json_file}: {num} example{'s' if num != 1 else ''}")
    if num:
        example = responses[0]
        response_text = example.get("response", str(example))
        print("Example response:")
        print(response_text)
    print("-" * 60)

# ---------------------------
# Main Procedure: Run Evaluation & Inference
# ---------------------------
if __name__ == "__main__":
    # --- Define model names for clarity ---
    baseline_name = "LLaMa‑3 Baseline (adapter‑merged)"
    modified_name = "LLaMa‑3 Modified (LoRA + Probes)"

    # --- Load Models ---
    print("Loading baseline model …")
    baseline_model, tokenizer = load_baseline()
    print("Loading modified model …")
    modified_model, _ = load_modified()  # using the same tokenizer

    # --- Generate Outputs ---
    baseline_output_file = "baseline.json"
    modified_output_file = "modified.json"
    print(f"Generating responses for {baseline_name} and saving to {baseline_output_file} …")
    _ = generate_responses(baseline_model, tokenizer, baseline_output_file)
    print(f"Generating responses for {modified_name} and saving to {modified_output_file} …")
    _ = generate_responses(modified_model, tokenizer, modified_output_file)

    # --- Print Summaries of Generated Responses ---
    print("\nResponse File Summaries:")
    print_response_summary(baseline_output_file, baseline_name)
    print_response_summary(modified_output_file, modified_name)

    # --- (Optional) Run Further Analysis ---
    # If you have further analysis steps relying on these outputs, run them here.
    # For example:
    print("Running analysis … (Note: analyze() may print results internally)")
    results = analyze(
        outputs_1=modified_output_file,  # Responses from modified model
        outputs_2=baseline_output_file,  # Responses from baseline model
        annotators_config="alpaca_eval_gpt4",
        openai_api_key=OPENAI_KEY,
        max_instances=805  # adjust as needed
    )
    if results is not None:
        header = f"{'Evaluator':<40}{'Baseline Score':>20}{'Modified Score':>20}"
        print("\nFinal Evaluation Results:")
        print(header)
        print("-" * len(header))
        print(results)
    else:
        print("\nAnalysis complete. (Results were printed by analyze() above.)")
