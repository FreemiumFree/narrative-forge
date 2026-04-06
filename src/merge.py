"""Merge LoRA adapter weights into the base model."""
import os

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def run_merge(
    config_path: str = "configs/training_config.yaml",
    adapter_dir: str = None,
):
    """Merge LoRA adapter into base model and save."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["base_model"]
    merged_dir = cfg["export"]["merged_dir"]

    if adapter_dir is None:
        adapter_dir = os.path.join(cfg["training"]["output_dir"], "final_adapter")

    if not os.path.exists(adapter_dir):
        print(f"ERROR: No adapter found at {adapter_dir}")
        print("Run 'python cli.py train' first.")
        return

    print("=== Narrative Forge — Merge Adapter ===\n")

    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading base model in FP16 (this uses ~14GB RAM)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    print(f"Loading adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(model, adapter_dir)

    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    os.makedirs(merged_dir, exist_ok=True)
    print(f"Saving merged model to {merged_dir}...")
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    print(f"\nMerge complete! Merged model saved to {merged_dir}")
    print("Next step: python cli.py export")
