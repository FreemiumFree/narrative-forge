"""Narrative Forge CLI — wraps all pipeline commands."""
import subprocess
import sys
import os

COMMANDS = {
    "setup": "Install dependencies and verify GPU",
    "prepare": "Extract and process book text into training data",
    "craft-analyze": "Analyze books with Claude Opus to extract craft techniques",
    "craft-generate": "Generate original training examples from craft catalog",
    "train": "Run LoRA fine-tuning",
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
    """Run the book extraction and chunking pipeline (Phase 1)."""
    import yaml
    sys.path.insert(0, os.path.dirname(__file__))
    from src.extract import extract_from_directory
    from src.chunk import chunk_text, classify_chunk
    import json

    with open("configs/data_config.yaml") as f:
        data_cfg = yaml.safe_load(f)

    raw_dir = data_cfg["raw_dir"]
    output_dir = data_cfg["processed_dir"]
    min_words = data_cfg["chunking"]["min_chunk_words"]
    max_words = data_cfg["chunking"]["max_chunk_words"]

    print("=== Phase 1: Extract and Chunk Books ===\n")
    print(f"Extracting text from {raw_dir}...")
    books = extract_from_directory(raw_dir)
    print(f"  Found {len(books)} book(s)")

    if not books:
        print("  WARNING: No books found. Add files to data/raw/")
        return

    all_chunks = []
    for book in books:
        chunks = chunk_text(book["text"], min_words, max_words)
        for chunk in chunks:
            chunk["type"] = classify_chunk(chunk["text"])
            chunk["source"] = book["source"]
        all_chunks.extend(chunks)
        print(f"  {book['source']}: {len(chunks)} chunks")

    print(f"  Total chunks: {len(all_chunks)}")

    os.makedirs(output_dir, exist_ok=True)
    chunks_path = os.path.join(output_dir, "chunks.jsonl")
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nDone! {len(all_chunks)} chunks saved to {chunks_path}")
    print("Next step: python cli.py craft-analyze")


def run_craft_analyze_cmd():
    """Analyze chunks with Claude Opus to extract craft techniques."""
    import json
    sys.path.insert(0, os.path.dirname(__file__))
    from src.craft_analyzer import analyze_chunks

    chunks_path = os.path.join("data", "processed", "chunks.jsonl")
    if not os.path.exists(chunks_path):
        print("ERROR: No chunks found. Run 'python cli.py prepare' first.")
        sys.exit(1)

    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    print(f"=== Phase 2: Craft Analysis with Claude Opus ===\n")
    print(f"  Loaded {len(chunks)} chunks")

    analysis_path = os.path.join("data", "processed", "craft_analysis.jsonl")
    analyses = analyze_chunks(chunks, analysis_path)

    print(f"\nDone! {len(analyses)} craft techniques cataloged")
    print(f"  Saved to {analysis_path}")
    print("Next step: python cli.py craft-generate")


def run_craft_generate_cmd():
    """Generate original training examples from craft analysis."""
    import json
    import random
    sys.path.insert(0, os.path.dirname(__file__))
    from src.craft_generator import generate_training_data

    analysis_path = os.path.join("data", "processed", "craft_analysis.jsonl")
    if not os.path.exists(analysis_path):
        print("ERROR: No craft analysis found. Run 'python cli.py craft-analyze' first.")
        sys.exit(1)

    analyses = []
    with open(analysis_path, encoding="utf-8") as f:
        for line in f:
            analyses.append(json.loads(line))

    print(f"=== Phase 3: Generate Original Training Examples ===\n")
    print(f"  Loaded {len(analyses)} craft techniques")

    generated_path = os.path.join("data", "processed", "craft_examples.jsonl")
    examples = generate_training_data(analyses, generated_path, examples_per_technique=3)

    # Split into train/val
    random.seed(42)
    random.shuffle(examples)
    split_idx = int(len(examples) * 0.9)
    train = examples[:split_idx]
    val = examples[split_idx:]

    train_path = os.path.join("data", "processed", "train.jsonl")
    val_path = os.path.join("data", "processed", "val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nDone! {len(examples)} original training examples generated")
    print(f"  Train: {len(train)} -> {train_path}")
    print(f"  Val: {len(val)} -> {val_path}")
    print("Next step: python cli.py train")


def run_train_cmd():
    """Run LoRA fine-tuning."""
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
        print("\nFull pipeline:")
        print("  1. prepare        Extract and chunk books")
        print("  2. craft-analyze  Analyze craft techniques (Claude Opus)")
        print("  3. craft-generate Generate original training examples (Claude Opus)")
        print("  4. train          Fine-tune the model")
        print("  5. evaluate       Review sample outputs")
        print("  6. merge → export → register  Deploy to Ollama")
        sys.exit(1)

    command = sys.argv[1]

    if command == "setup":
        run_setup()
    elif command == "prepare":
        run_prepare_cmd()
    elif command == "craft-analyze":
        run_craft_analyze_cmd()
    elif command == "craft-generate":
        run_craft_generate_cmd()
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
