"""Narrative Forge CLI — wraps all pipeline commands."""
import subprocess
import sys
import os

COMMANDS = {
    "setup": "Install dependencies and verify GPU",
    "prepare": "Extract and process book text into training data",
    "train": "Run QLoRA fine-tuning",
    "evaluate": "Generate sample outputs from the trained model",
    "merge": "Merge LoRA adapter into base model",
    "export": "Convert merged model to GGUF format",
    "register": "Register model with Ollama",
}


def run_setup():
    """Run the setup script for the current platform."""
    if os.name == "nt":
        git_bash = r"C:\Program Files\Git\bin\bash.exe"
        if os.path.exists(git_bash):
            subprocess.run([git_bash, "setup.sh"], check=True)
        else:
            subprocess.run(["cmd", "/c", "setup.bat"], check=True)
    else:
        subprocess.run(["bash", "setup.sh"], check=True)


def run_prepare_cmd():
    """Run the data preparation pipeline."""
    import yaml
    sys.path.insert(0, os.path.dirname(__file__))
    from src.prepare import run_prepare, load_templates

    with open("configs/data_config.yaml") as f:
        data_cfg = yaml.safe_load(f)

    templates = load_templates(data_cfg["templates_file"])

    stats = run_prepare(
        raw_dir=data_cfg["raw_dir"],
        output_dir=data_cfg["processed_dir"],
        templates=templates,
        min_chunk_words=data_cfg["chunking"]["min_chunk_words"],
        max_chunk_words=data_cfg["chunking"]["max_chunk_words"],
        train_split=data_cfg["train_split"],
    )

    print(f"\nDone! {stats['total_pairs']} training pairs from {stats['total_books']} book(s).")
    print(f"Type breakdown: {stats['type_breakdown']}")


def run_train_cmd():
    """Run QLoRA fine-tuning."""
    sys.path.insert(0, os.path.dirname(__file__))
    from src.train import run_training
    run_training()


def run_evaluate_cmd():
    """Generate sample outputs for review."""
    sys.path.insert(0, os.path.dirname(__file__))
    from src.evaluate import run_evaluation
    run_evaluation()


def run_merge_cmd():
    """Merge LoRA adapter into base model."""
    sys.path.insert(0, os.path.dirname(__file__))
    from src.merge import run_merge
    run_merge()


def run_export_cmd():
    """Convert merged model to GGUF."""
    sys.path.insert(0, os.path.dirname(__file__))
    from src.export import run_export
    run_export()


def run_register_cmd():
    """Register the model with Ollama."""
    import yaml

    with open("configs/training_config.yaml") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["export"]["ollama_model_name"]
    gguf_path = os.path.join(cfg["export"]["gguf_dir"], f"{model_name}.gguf")

    if not os.path.exists(gguf_path):
        print(f"ERROR: No GGUF model found at {gguf_path}")
        print("Run 'python cli.py export' first.")
        sys.exit(1)

    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: Ollama is not installed.")
        print("Install from: https://ollama.com/download")
        print("After installing, run this command again.")
        sys.exit(1)

    print(f"=== Registering {model_name} with Ollama ===\n")

    result = subprocess.run(
        ["ollama", "create", model_name, "-f", "Modelfile"],
        capture_output=False,
    )

    if result.returncode == 0:
        print(f"\nModel '{model_name}' registered with Ollama!")
        print(f"\nTest it:  ollama run {model_name}")
        print(f"API:     curl http://localhost:11434/api/generate -d '{{\"model\": \"{model_name}\", \"prompt\": \"Describe an ancient forest at dawn.\"}}'")
    else:
        print("\nERROR: Failed to register model with Ollama.")
        sys.exit(1)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Narrative Forge — LLM Fine-Tuning Pipeline\n")
        print("Usage: python cli.py <command>\n")
        print("Commands:")
        for cmd, desc in COMMANDS.items():
            print(f"  {cmd:12s} {desc}")
        sys.exit(1)

    command = sys.argv[1]

    if command == "setup":
        run_setup()
    elif command == "prepare":
        run_prepare_cmd()
    elif command == "train":
        run_train_cmd()
    elif command == "evaluate":
        run_evaluate_cmd()
    elif command == "merge":
        run_merge_cmd()
    elif command == "export":
        run_export_cmd()
    elif command == "register":
        run_register_cmd()
    else:
        print(f"Command '{command}' not yet implemented.")
        sys.exit(1)


if __name__ == "__main__":
    main()
