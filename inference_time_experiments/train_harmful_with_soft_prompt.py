import gc
import pickle
import time
from pathlib import Path
from typing import List

import hydra
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Import existing modules from your project
from evaluate import evaluate_metric, evaluate_model, evaluate_model_and_metric_harmful
from obf_reps.config import ExperimentConfig
from obf_reps.data import ObfusDataModule, ObfusDataset
from obf_reps.logging import Logger
from obf_reps.metrics import ObfMetric, TestCachedReps
from obf_reps.models import ModelBase
from obf_reps.optimize.loss import LossFunctionBase
from obf_reps.optimize.optimizers import OptimizerBase
from obf_reps.plotting import plot_lat_scan
from obf_reps.utils import (
    convert_path,
    get_test_cached_reps,
    validate_and_create_experiment_config,
)

# Import the soft prompt components from basic_science
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn

class SoftPrompt(nn.Module):
    def __init__(self, num_virtual_tokens, hidden_dim):
        super().__init__()
        self.soft_tokens = nn.Parameter(torch.randn(num_virtual_tokens, hidden_dim))

    def forward(self, input_embeds):
        batch_size = input_embeds.size(0)
        # expand: shape = [batch_size, num_virtual_tokens, hidden_dim]
        expanded = self.soft_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([expanded, input_embeds], dim=1)


class SoftPromptWrapper(nn.Module):
    def __init__(self, model, soft_prompt_module, tokenizer):
        super().__init__()
        self.model = model
        self.soft_prompt_module = soft_prompt_module
        self.tokenizer = tokenizer
        self.input_embedding_layer = model.get_input_embeddings()
        # Ensure that the following dtype matches model parameters.
        self.model_dtype = next(model.parameters()).dtype

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get the standard token embeddings.
        real_embeds = self.input_embedding_layer(input_ids)
        # Pass them through the soft-prompt module.
        full_embeds = self.soft_prompt_module(real_embeds).to(self.model_dtype)

        # Adjust attention mask: prepend ones for soft tokens.
        if attention_mask is not None:
            batch_size, seq_len = attention_mask.shape
            prefix_mask = torch.ones(
                batch_size,
                self.soft_prompt_module.soft_tokens.shape[0],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            extended_mask = None

        outputs = self.model(
            inputs_embeds=full_embeds,
            attention_mask=extended_mask,
            labels=labels,
        )
        return outputs


def enforce_random_orthant(soft_prompt_flat, chosen_indices, chosen_signs, min_magnitude=1e-4):
    # Ensures that the elements at indices in chosen_indices have the signs in chosen_signs.
    with torch.no_grad():
        chosen_data = soft_prompt_flat[chosen_indices]
        abs_values = torch.maximum(
            chosen_data.abs(),
            torch.tensor(min_magnitude, device=chosen_data.device)
        )
        # Force these coordinates to have the target signs.
        forced = (abs_values * chosen_signs).to(soft_prompt_flat.dtype)
        soft_prompt_flat[chosen_indices] = forced


##################################################################
# Integration initialization:
##################################################################
def init_model_with_soft_prompt(experiment_cfg):
    """
    Wrap the existing model with a soft prompt module.
    We assume that experiment_cfg.optimizer.model is the base model.
    """
    base_model = experiment_cfg.optimizer.model

    # Get the hidden dim from the model's input embeddings.
    hidden_dim = base_model.get_input_embeddings().weight.shape[1]
    # Get number of virtual tokens from config - default to 20 if not provided.
    num_virtual_tokens = experiment_cfg.get("soft_prompt_cfg", {}).get("num_virtual_tokens", 20)

    # Create the soft prompt module.
    sp_module = SoftPrompt(num_virtual_tokens, hidden_dim).to(base_model.device)

    # Prepare target signs for enforcing the orthant.
    # We flatten the soft tokens and record their initial sign.
    target_signs = sp_module.soft_tokens.detach().sign().clone().flatten()

    # Add soft_prompt_cfg to config if it doesn't exist:
    if "soft_prompt_cfg" not in experiment_cfg:
        experiment_cfg.soft_prompt_cfg = {}

    # Set default values if not provided.
    experiment_cfg.soft_prompt_cfg.setdefault("initial_active", 5)
    experiment_cfg.soft_prompt_cfg.setdefault("increment", 5)
    experiment_cfg.soft_prompt_cfg.setdefault("num_virtual_tokens", num_virtual_tokens)
    experiment_cfg.soft_prompt_cfg.target_signs = target_signs

    # For the tokenizer, you may use one specified in the config.
    # Here we try to initialize one from the model name, assuming experiment_cfg.model_name is set.
    if "tokenizer" in experiment_cfg:
        tokenizer = experiment_cfg.tokenizer
    else:
        # Fallback: you might want to add a model_name field to your config.
        model_name = getattr(experiment_cfg, "model_name", "meta-llama/Meta-Llama-3-8B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        experiment_cfg.tokenizer = tokenizer

    # Wrap the model.
    wrapped_model = SoftPromptWrapper(base_model, sp_module, tokenizer).to(base_model.device)

    # Replace the optimizer's model with the wrapped one.
    experiment_cfg.optimizer.model = wrapped_model

    # Return the soft prompt module so that further training can update it.
    return sp_module


##################################################################
# Trainer loop with integrated soft prompt
##################################################################
def train_attack(
    optimizer: OptimizerBase,
    metric: ObfMetric,
    epochs: int,
    train_dataloader: DataLoader,
    experiment_cfg: ExperimentConfig,
    attack_defense_epochs: int,
):
    """
    Alternate between defender evaluation and attacker updates. During the attacker
    phase we update the soft prompt parameters for an increasing number of coordinates,
    enforcing that those coordinates live in a target orthant.
    """
    test_cached_reps: List[TestCachedReps] = []
    params_last_train_epoch: int = -1

    # Set up the soft prompt active coordinate counter.
    active_coords = experiment_cfg.soft_prompt_cfg.initial_active
    # Get the total number of coordinates available in soft prompt.
    sp_flat = optimizer.model.soft_prompt_module.soft_tokens.view(-1)
    total_coords = sp_flat.numel()
    target_signs = experiment_cfg.soft_prompt_cfg.target_signs

    for at_def_epoch in range(attack_defense_epochs):

        start_epoch_time = time.perf_counter()

        # Defender phase evaluation
        obfus_reps_with_behavior = evaluate_model_and_metric_harmful(
            model=optimizer.model,
            obfus_data_module=experiment_cfg.obfus_data_module,
            concept_data_module=experiment_cfg.concept_data_module,
            metric=metric,
            logger=experiment_cfg.logger,
            at_def_epoch=at_def_epoch,
            who_just_went="defender",
            gen_len=experiment_cfg.gen_len,
            params_last_train_epoch=params_last_train_epoch,
            evaluate_metric_on_gens=False,
            evaluate_behavior=False,
            cached_reps=test_cached_reps,
            batch_size=experiment_cfg.eval_batch_size,
        )

        if at_def_epoch == 0:
            test_cached_reps.append(
                TestCachedReps(
                    label="random-obf",
                    reps=obfus_reps_with_behavior,
                    attack_defence_epoch=-1,
                )
            )

        if experiment_cfg.save_probe_path:
            path = convert_path(experiment_cfg.save_probe_path, at_def_epoch)
            # Assert that the file doesn't exist to avoid overwriting.
            assert not path.exists(), f"File already exists at {path}. Choose a different path."
            metric.save_probe(path)

        # Attacker turn
        loaded_params = False
        if experiment_cfg.load_tunable_params_path:
            path = convert_path(experiment_cfg.load_tunable_params_path, at_def_epoch)
            if path.exists():
                with open(path, "rb") as f:
                    loaded_params = pickle.load(f)
                # Assuming that loaded_params refer to the soft prompt tokens.
                optimizer.model.soft_prompt_module.soft_tokens = loaded_params
                print(f"Loaded tunable parameters from {path}")
                loaded_params = True

        if not loaded_params:
            if experiment_cfg.reinit_tunable_params:
                optimizer.model.soft_prompt_module.soft_tokens.data.normal_()

            print("\n" + "="*56)
            print(f"Training attack defence {at_def_epoch} on first {active_coords} coordinates")
            print("="*56 + "\n")
            for epoch in tqdm.tqdm(range(epochs)):
                for batch in tqdm.tqdm(train_dataloader, leave=False):
                    optimizer.step(batch)
                    # After each step enforce the orthant on the active coordinates.
                    sp_tokens = optimizer.model.soft_prompt_module.soft_tokens
                    sp_flat = sp_tokens.view(-1)
                    chosen_indices = torch.arange(active_coords, device=sp_flat.device)
                    enforce_random_orthant(
                        sp_flat, chosen_indices, target_signs[chosen_indices],
                        min_magnitude=1e-4
                    )

            torch.cuda.empty_cache()
            if experiment_cfg.save_tunable_params_path:
                path = convert_path(experiment_cfg.save_tunable_params_path, at_def_epoch)
                assert not path.exists(), f"File already exists at {str(path)}. Choose a different path."
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    pickle.dump(optimizer.model.soft_prompt_module.soft_tokens, f)
                print(f"Tunable soft prompt parameters saved to {str(path)}")

        params_last_train_epoch += 1

        # Attacker evaluation
        obfus_reps_with_behavior = evaluate_model_and_metric_harmful(
            model=optimizer.model,
            obfus_data_module=experiment_cfg.obfus_data_module,
            concept_data_module=experiment_cfg.concept_data_module,
            metric=metric,
            logger=experiment_cfg.logger,
            at_def_epoch=at_def_epoch,
            who_just_went="attacker",
            gen_len=experiment_cfg.gen_len,
            params_last_train_epoch=params_last_train_epoch,
            evaluate_metric_on_gens=False,
            evaluate_behavior=True,
            cached_reps=test_cached_reps,
            batch_size=experiment_cfg.eval_batch_size,
        )
        test_cached_reps.append(
            TestCachedReps(
                label=f"attacker-{at_def_epoch}",
                reps=obfus_reps_with_behavior,
                attack_defence_epoch=at_def_epoch,
            )
        )

        # Increase the number of active soft prompt coordinates.
        active_coords = min(active_coords + experiment_cfg.soft_prompt_cfg.increment, total_coords)

        # Defender refitting stage (if needed)
        if at_def_epoch < attack_defense_epochs - 1:
            print("Refitting probe")
            start = time.time()
            metric.refit(at_def_epoch)  # type: ignore
            end = time.time()
            print(f"Elapsed time for refit at_def_epoch {at_def_epoch}: {end - start} seconds")

        end_epoch_time = time.perf_counter()
        print("\n\n" + "#" * 30)
        print(f"MARK: Epoch {at_def_epoch} took {end_epoch_time - start_epoch_time:.4f} seconds")
        print("#" * 30 + "\n\n")


##################################################################
# Main entry point via Hydra.
##################################################################
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    experiment_cfg: ExperimentConfig = validate_and_create_experiment_config(cfg)

    # Log dataset sizes.
    experiment_cfg.logger.log({"concept_dataset_train_size": len(experiment_cfg.concept_data_module.train_dataset)})
    experiment_cfg.logger.log({"concept_dataset_val_size": len(experiment_cfg.concept_data_module.val_dataset)})
    experiment_cfg.logger.log({"concept_dataset_test_size": len(experiment_cfg.concept_data_module.test_dataset)})

    if experiment_cfg.save_pca_path:
        experiment_cfg.metric.save_pca(experiment_cfg.save_pca_path)

    # If training metric-only, do not perform soft prompt training.
    if experiment_cfg.train_metric_only:
        if experiment_cfg.save_probe_path:
            path = convert_path(experiment_cfg.save_probe_path, attack_defense_epochs=0)
            assert not path.exists(), f"File already exists at {str(path)}. Choose different path."
            experiment_cfg.metric.save_probe(path)
        evaluate_model_and_metric_harmful(
            model=experiment_cfg.optimizer.model,
            obfus_data_module=experiment_cfg.obfus_data_module,
            concept_data_module=experiment_cfg.concept_data_module,
            metric=experiment_cfg.metric,
            logger=experiment_cfg.logger,
            at_def_epoch=1,
            who_just_went="defender",
            gen_len=experiment_cfg.gen_len,
            evaluate_metric_on_gens=False,
            evaluate_behavior=False,
            batch_size=experiment_cfg.eval_batch_size,
            params_last_train_epoch=0,
        )
        return

    # Initialize the soft prompt wrapper.
    _ = init_model_with_soft_prompt(experiment_cfg)

    # Call the integrated training routine.
    _ = train_attack(
        optimizer=experiment_cfg.optimizer,
        metric=experiment_cfg.metric,
        epochs=experiment_cfg.epochs,
        train_dataloader=experiment_cfg.obfus_data_module.train_dataloader,  # type: ignore
        experiment_cfg=experiment_cfg,
        attack_defense_epochs=experiment_cfg.attack_defense_epochs,
    )


if __name__ == "__main__":
    main()
