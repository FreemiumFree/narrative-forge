"""QLoRA fine-tuning script for narrative generation."""
import os

import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


def format_training_example(example: dict, tokenizer) -> str:
    """Format an instruction/completion pair for training."""
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def run_training(config_path: str = "configs/training_config.yaml"):
    """Run LoRA fine-tuning."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("=== Narrative Forge — LoRA Training ===\n")

    model_name = cfg["base_model"]
    print(f"Base model: {model_name}")

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available. Training requires a GPU.")
        return
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {vram_gb:.1f} GB\n")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model in FP16 (no quantization — 7B in FP16 = ~14GB, fits in 24GB VRAM)
    print("Loading model in FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )

    model = get_peft_model(model, peft_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)\n")

    print("Loading training data...")
    data_cfg_path = "configs/data_config.yaml"
    with open(data_cfg_path) as f:
        data_cfg = yaml.safe_load(f)

    processed_dir = data_cfg["processed_dir"]
    dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(processed_dir, "train.jsonl"),
            "validation": os.path.join(processed_dir, "val.jsonl"),
        },
    )
    print(f"Train examples: {len(dataset['train'])}")
    print(f"Validation examples: {len(dataset['validation'])}\n")

    train_cfg = cfg["training"]

    # Calculate warmup steps from ratio
    total_steps = (
        len(dataset["train"]) // (train_cfg["per_device_train_batch_size"] * train_cfg["gradient_accumulation_steps"])
    ) * train_cfg["num_train_epochs"]
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])

    sft_config = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_steps=warmup_steps,
        fp16=True,
        bf16=False,
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        optim="adamw_torch",
        seed=train_cfg["seed"],
        report_to="none",
        remove_unused_columns=False,
        max_length=train_cfg["max_seq_length"],
        gradient_checkpointing=True,
    )

    # Pass the already-wrapped PEFT model — no peft_config here
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        formatting_func=lambda ex: format_training_example(ex, tokenizer),
    )

    print("Starting training...\n")
    trainer.train()

    final_adapter_dir = os.path.join(train_cfg["output_dir"], "final_adapter")
    trainer.save_model(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print(f"\nTraining complete! Adapter saved to {final_adapter_dir}")
    print("Next step: python cli.py merge")
