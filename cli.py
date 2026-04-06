"""Narrative Forge CLI — wraps all pipeline commands."""
import sys

COMMANDS = {
    "setup": "Install dependencies and verify GPU",
    "prepare": "Extract and process book text into training data",
    "train": "Run QLoRA fine-tuning",
    "evaluate": "Generate sample outputs from the trained model",
    "merge": "Merge LoRA adapter into base model",
    "export": "Convert merged model to GGUF format",
    "register": "Register model with Ollama",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Narrative Forge — LLM Fine-Tuning Pipeline\n")
        print("Usage: python cli.py <command>\n")
        print("Commands:")
        for cmd, desc in COMMANDS.items():
            print(f"  {cmd:12s} {desc}")
        sys.exit(1)

    command = sys.argv[1]
    print(f"Command '{command}' not yet implemented.")
    sys.exit(1)


if __name__ == "__main__":
    main()
