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
    else:
        print(f"Command '{command}' not yet implemented.")
        sys.exit(1)


if __name__ == "__main__":
    main()
