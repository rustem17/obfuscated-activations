#!/usr/bin/env python3
import os
import sys
import json
import torch
import subprocess
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoTokenizer, GenerationConfig


# --- Local imports ---
from src.probe_evals import *
from src.probe_archs import *
from src.probe_training import *
from src import EleutherSparseAutoencoder


# Configuration
os.environ["OPENAI_API_KEY"] = "your-api-key"  # Replace with your key
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "Mechanistic-Anomaly-Detection/llama3-oat-generation-linear"
LOCAL_DIR = Path("llama3-oat-generation-linear")

# Path setup
CONVERSATION_DIR = Path("conversations")
RESULTS_DIR = Path("results")
EVALUATION_DIR = Path("data/mt_bench/model_answer")

def load_model(instruct=True):
    """Load model with error handling and PEFT support"""
    try:
        snapshot_download(repo_id=REPO_ID, local_dir=LOCAL_DIR)
        encoder = EleutherSparseAutoencoder.load_llama3_sae(None, instruct=instruct)

        if isinstance(encoder.model, PeftModel):
            encoder.model = encoder.model.merge_and_unload()

        encoder.model.to(DEVICE).eval()
        return encoder.model, encoder.tokenizer

    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        sys.exit(1)

def generate_and_save_conversations(model, tokenizer, model_id, resume=False):
    """Generate and save conversations with resume support"""
    CONVERSATION_DIR.mkdir(exist_ok=True, parents=True)
    conv_path = CONVERSATION_DIR / f"{model_id}.jsonl"

    dataset = load_dataset("lighteval/mt-bench", split="train")
    existing_ids = set()

    if resume and conv_path.exists():
        print(f"Resuming from existing conversations at {conv_path}")
        with open(conv_path) as f:
            for line in f:
                existing_ids.add(json.loads(line)["question_id"])

    with open(conv_path, "a" if resume else "w") as f:
        for q in dataset:
            if q["question_id"] in existing_ids:
                continue

            conversation = []
            record = {
                "question_id": q["question_id"],
                "turns": [],
                "responses": [],
                "full_history": []
            }

            try:
                for turn in q["turns"]:
                    # Build conversation context
                    messages = [{"role": "user", "content": turn}]

                    # Format with model's chat template
                    inputs = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).to(DEVICE)

                    # Generate response
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=512,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        do_sample=True
                    )

                    # Decode and process response
                    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    assistant_response = full_response.split("assistant\n")[-1].strip()

                    # Save conversation state
                    record["turns"].append(turn)
                    record["responses"].append(assistant_response)
                    record["full_history"].append({
                        "user_input": turn,
                        "model_response": assistant_response
                    })

                    # Update conversation for next turn
                    messages.append({"role": "assistant", "content": assistant_response})

                # Save completed conversation
                f.write(json.dumps(record) + "\n")
                f.flush()  # Ensure partial writes are saved

            except Exception as e:
                print(f"Error processing question {q['question_id']}: {str(e)}")
                continue

    return conv_path

def convert_to_fastchat_format(conv_path, model_id):
    """Convert saved conversations to FastChat evaluation format"""
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVALUATION_DIR / f"{model_id}.jsonl"

    with open(conv_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            conv = json.loads(line)
            f_out.write(json.dumps({
                "question_id": conv["question_id"],
                "turns": conv["responses"]
            }) + "\n")

    return output_path

def run_evaluation(model_id):
    """Execute FastChat evaluation pipeline"""
    try:
        # Generate judgments
        subprocess.run([
            "python", "-m", "fastchat.llm_judge.gen_judgment",
            "--model-list", model_id,
            "--bench-name", "mt-bench"
        ], check=True)

        # Get results
        result = subprocess.run([
            "python", "-m", "fastchat.llm_judge.show_result",
            "--bench-name", "mt-bench"
        ], capture_output=True, text=True, check=True)

        return parse_results(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed: {str(e)}")
        return None

def parse_results(output):
    """Parse FastChat's evaluation results"""
    results = {}
    for line in output.split("\n"):
        if "Turn 1" in line:
            results["first_turn"] = float(line.split()[-1])
        elif "Turn 2" in line:
            results["second_turn"] = float(line.split()[-1])
        elif "Average" in line:
            results["average"] = float(line.split()[-1])
    return results

def main(model_id, resume=False):
    """Complete workflow: generation -> conversion -> evaluation"""
    # Step 1: Generate or load conversations
    conv_path = CONVERSATION_DIR / f"{model_id}.jsonl"
    # if not conv_path.exists() or not resume:
    #     print(f"Generating responses for {model_id}...")
    #     model, tokenizer = load_model()
    #     generate_and_save_conversations(model, tokenizer, model_id, resume)

    # # Step 2: Convert to FastChat format
    # print("Converting conversations...")
    # eval_file = convert_to_fastchat_format(conv_path, model_id)

    # Step 3: Run evaluation
    print("Running evaluation...")
    results = run_evaluation(model_id)

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MT-Bench Evaluation Pipeline")
    parser.add_argument("--model-id", required=True, help="Unique model identifier")
    parser.add_argument("--resume", action="store_true", help="Resume generation")
    args = parser.parse_args()

    # Execute complete pipeline
    results = main(args.model_id, args.resume)

    # Display results
    if results:
        print("\n=== Evaluation Results ===")
        print(f"First Turn Score: {results.get('first_turn', 0):.2f}")
        print(f"Second Turn Score: {results.get('second_turn', 0):.2f}")
        print(f"Average Score: {results.get('average', 0):.2f}")
    else:
        print("Evaluation failed - check error logs")
