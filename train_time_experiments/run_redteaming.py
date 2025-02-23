import argparse
import json
import os

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from automated_redteaming import *
from src import *
from src.probe_evals import *
from src.probe_archs import *
from src.probe_training import *

load_dotenv()
hf_access_token = os.getenv("HUGGINGFACE_API_KEY")
openai_api_key = os.environ["OPENAI_API_KEY"]

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_name", type=str, help="Name of the file to process")
    parser.add_argument("--model_type", type=str, choices=["llama3", "gemma2"], default="llama3")
    parser.add_argument("--local_dir", type=str, help="Path to local weights directory")    parser.add_argument("--device", type=str, default="cuda", help="Name of device")
    parser.add_argument("--attack_type", type=str, default="pair", choices=["pair", "gcg"], help="Name of attack")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use (e.g. 0, 1, etc)")

    return parser


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.model_type == "llama3":
        encoder = EleutherSparseAutoencoder.load_llama3_sae(None, instruct=True)
    else:
        encoder = DeepmindSparseAutoencoder.load_gemma2_sae(None, 11)

    if not isinstance(encoder.model, PeftModel):
        encoder.model = PeftModel.from_pretrained(
            encoder.model, args.local_dir
        )
        encoder.model = encoder.model.merge_and_unload()

    linear_probes = load_probes(
        os.path.join(args.local_dir, "probes.pt")
    )

    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_name,
    #     token=hf_access_token,
    #     torch_dtype=torch.bfloat16
    # ).to(args.device)

    # tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.padding_side = "left"

    dataset = load_dataset("abhayesian/augmented_advbench_v5")["train"].shuffle()

    if args.attack_type == "pair":
        pair = PromptAutomaticIterativeRefinement(
            model=encoder.model,
            tokenizer=encoder.tokenizer,
            api_key=openai_api_key,
        )
    elif args.attack_type == "gcg":
        gcg = GreedyCoordinateGradient(
            model=encoder.model,
            tokenizer=encoder.tokenizer,
        )

    if os.path.exists(args.file_name):
        with open(args.file_name, "r") as f:
            results = json.load(f)
    else:
        results = []

    # for i in range(len(dataset)):
    for i in range(10):
        try:
            prompt = dataset[i]["prompt"]
            target = " ".join(dataset[i]["rejected"].split()[:10])
            chosen_response = dataset[i]["chosen"]

            if args.attack_type == "pair":
                # Get PAIR attack
                top_candidate, high_score, response = pair.compute_adversary(prompt, target)
                attack_dict = {
                    "prompt": prompt,
                    "refusal": chosen_response,
                    "jailbreak": top_candidate,
                    "model_response": response,
                    "pair_score": high_score,
                }
            elif args.attack_type == "gcg":
                top_candidate, gcg_loss, response = gcg.compute_adversary(prompt, target)
                attack_dict = {
                    "prompt": prompt,
                    "refusal": chosen_response,
                    "jailbreak": top_candidate,
                    "model_response": response,
                    "gcg_loss": gcg_loss,
                }
            elif args.attack_type == "msj":
                pass

            # Get StrongReject response
            score = autograde_response(prompt, response).score_zero_to_one
            if score == 0: # If refusal
                continue
            attack_dict["strongreject_score"] = score

            # Add new example to the file
            results.append(attack_dict)
            with open(args.file_name, "w") as f:
                json.dump(results, f, indent=4)

        except Exception as e:
            raise e


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
