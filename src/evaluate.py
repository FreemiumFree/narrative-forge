"""Generate sample outputs from a trained model for manual quality review."""
import os
import json
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import yaml

EVAL_PROMPTS = [
    {
        "type": "prose",
        "prompt": "Write a vivid description of an ancient temple hidden deep in a mountain forest, discovered at dawn by a lone traveler.",
    },
    {
        "type": "dialogue",
        "prompt": "Write a tense dialogue between a young hero who has just learned their mentor has been lying to them about the source of their power.",
    },
    {
        "type": "action",
        "prompt": "Write an intense combat scene between two warriors fighting on a narrow stone bridge above a chasm during a thunderstorm.",
    },
    {
        "type": "worldbuilding",
        "prompt": "Describe the magic system of a world where power flows from divine bonds between mortals and gods, and where each god's domain governs a specific aspect of reality.",
    },
    {
        "type": "internal",
        "prompt": "Write the internal monologue of a hero standing at the edge of a battlefield, realizing that the war they fought was built on a lie.",
    },
    {
        "type": "pacing",
        "prompt": "Write a chapter opening that transitions from the aftermath of a devastating battle to a quiet moment of reflection between two survivors.",
    },
]


def run_evaluation(
    config_path: str = "configs/training_config.yaml",
    adapter_dir: str = None,
    output_dir: str = "models/samples",
):
    """Generate sample outputs from the trained model."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["base_model"]
    if adapter_dir is None:
        adapter_dir = os.path.join(cfg["training"]["output_dir"], "final_adapter")

    if not os.path.exists(adapter_dir):
        print(f"ERROR: No adapter found at {adapter_dir}")
        print("Run 'python cli.py train' first.")
        return

    print("=== Narrative Forge — Evaluation ===\n")

    quant_cfg = cfg["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )

    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print(f"Loading adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples = []

    for i, eval_prompt in enumerate(EVAL_PROMPTS):
        print(f"\nGenerating sample {i + 1}/{len(EVAL_PROMPTS)} ({eval_prompt['type']})...")

        messages = [{"role": "user", "content": eval_prompt["prompt"]}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        samples.append({
            "type": eval_prompt["type"],
            "prompt": eval_prompt["prompt"],
            "response": response,
        })

        print(f"  {response[:100]}...")

    output_path = os.path.join(output_dir, f"eval_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    md_path = os.path.join(output_dir, f"eval_{timestamp}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Narrative Forge — Evaluation Samples\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Model:** {model_name}\n")
        f.write(f"**Adapter:** {adapter_dir}\n\n---\n\n")
        for sample in samples:
            f.write(f"## {sample['type'].title()}\n\n")
            f.write(f"**Prompt:** {sample['prompt']}\n\n")
            f.write(f"**Response:**\n\n{sample['response']}\n\n---\n\n")

    print(f"\nSamples saved to:")
    print(f"  JSON: {output_path}")
    print(f"  Markdown: {md_path}")
    print("\nReview the markdown file to assess quality.")
