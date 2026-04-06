# Narrative Forge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a QLoRA fine-tuning pipeline that trains Mistral 7B on fantasy novels and serves the result via Ollama for the fantasy-book-engine to call.

**Architecture:** Python CLI (`cli.py`) wraps all operations: data preparation, QLoRA training, adapter merging, GGUF export, and Ollama registration. Data flows from raw book files through a chunking/classification pipeline into instruction/completion JSONL pairs, which feed the trainer. The trained model is merged, quantized, and registered with Ollama for HTTP inference.

**Tech Stack:** Python 3.12, PyTorch (CUDA 13.1), Hugging Face Transformers, PEFT, bitsandbytes, trl, Ollama, llama.cpp (for GGUF conversion)

**System:** Windows 11, 2x NVIDIA TITAN RTX (24GB each), i9-7980XE, 128GB RAM, `make` not available — use `python cli.py <command>` instead.

---

## File Structure

```
narrative-forge/
├── cli.py                            # CLI entry point: python cli.py <command>
├── requirements.txt                  # All Python dependencies
├── setup.sh                          # Venv creation + dependency install (bash)
├── setup.bat                         # Same for Windows cmd
├── .gitignore                        # Ignore data/raw, models, .env, __pycache__
├── README.md                         # Setup + usage guide
├── configs/
│   ├── training_config.yaml          # Model, QLoRA, and training hyperparameters
│   └── data_config.yaml              # Chunk sizes, classification rules, templates path
├── data/
│   ├── raw/                          # User's book files (gitignored)
│   ├── processed/                    # Output JSONL + stats
│   └── templates/
│       └── default.yaml              # Instruction templates per chunk type
├── src/
│   ├── __init__.py
│   ├── extract.py                    # Text extraction from txt/epub/pdf
│   ├── chunk.py                      # Text chunking and narrative classification
│   ├── pair_generator.py             # Instruction/completion pair creation
│   ├── prepare.py                    # Orchestrates extract → chunk → pairs → JSONL
│   ├── train.py                      # QLoRA fine-tuning
│   ├── merge.py                      # Merge LoRA adapter into base model
│   ├── export.py                     # Convert merged model to GGUF
│   └── evaluate.py                   # Generate sample outputs for review
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Shared fixtures (sample text, temp dirs)
│   ├── test_extract.py               # Text extraction tests
│   ├── test_chunk.py                 # Chunking and classification tests
│   ├── test_pair_generator.py        # Training pair generation tests
│   └── test_prepare.py              # End-to-end data pipeline tests
├── models/                           # Training outputs (gitignored)
│   ├── checkpoints/
│   ├── merged/
│   ├── gguf/
│   └── samples/
└── Modelfile                         # Ollama model definition
```

---

## Task 1: Create GitHub Repository and Project Skeleton

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `cli.py` (stub)
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `configs/` directory
- Create: `data/raw/`, `data/processed/`, `data/templates/` directories
- Create: `models/` directory

- [ ] **Step 1: Create GitHub repo**

```bash
cd C:\Users\nhf56\Documents
mkdir narrative-forge
cd narrative-forge
git init
```

- [ ] **Step 2: Create .gitignore**

Create file `.gitignore`:

```
# Book files (copyright)
data/raw/

# Model weights (too large)
models/

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.eggs/
venv/
.venv/

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
Thumbs.db
.DS_Store

# Jupyter
.ipynb_checkpoints/
```

- [ ] **Step 3: Create directory structure**

```bash
mkdir -p src tests configs data/raw data/processed data/templates models/checkpoints models/merged models/gguf models/samples notebooks
touch src/__init__.py tests/__init__.py
```

- [ ] **Step 4: Create stub CLI**

Create file `cli.py`:

```python
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
```

- [ ] **Step 5: Create README.md**

Create file `README.md`:

```markdown
# Narrative Forge

Fine-tune open-source LLMs on your favorite fantasy novels to create a local creative writing engine.

Works in tandem with [fantasy-book-engine](../fantasy-book-engine) — this project generates all creative content (prose, dialogue, plot pacing), while the engine handles orchestration, world-bible consistency, and continuity.

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA support (tested on 2x TITAN RTX)
- Ollama (installed during setup)

## Quick Start

```bash
python cli.py setup       # Create venv, install deps, verify GPU
# Place your books in data/raw/ (txt, epub, pdf)
python cli.py prepare     # Process books into training data
python cli.py train       # QLoRA fine-tune (~1.5-3 hours)
python cli.py evaluate    # Generate samples for review
python cli.py merge       # Merge adapter into base model
python cli.py export      # Convert to GGUF
python cli.py register    # Register with Ollama
```

## Design Docs

- [Design Spec](docs/superpowers/specs/2026-04-06-narrative-forge-design.md)
- [Implementation Plan](docs/superpowers/plans/2026-04-06-narrative-forge.md)
```

- [ ] **Step 6: Create GitHub repo and push**

```bash
git add .
git commit -m "chore: initialize narrative-forge project skeleton"
gh repo create narrative-forge --public --source=. --push
```

---

## Task 2: Python Environment and Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `setup.sh`
- Create: `setup.bat`
- Modify: `cli.py` — wire up `setup` command

- [ ] **Step 1: Create requirements.txt**

Create file `requirements.txt`:

```
# Core ML
torch>=2.6.0
transformers>=4.48.0
peft>=0.14.0
bitsandbytes>=0.45.0
trl>=0.14.0
accelerate>=1.3.0
datasets>=3.2.0

# Data processing
ebooklib>=0.18
beautifulsoup4>=4.12.0
PyPDF2>=3.0.0
pyyaml>=6.0.0

# Evaluation
sentencepiece>=0.2.0

# Testing
pytest>=8.0.0

# GGUF conversion
huggingface_hub>=0.27.0
```

- [ ] **Step 2: Create setup.sh**

Create file `setup.sh`:

```bash
#!/bin/bash
set -e

echo "=== Narrative Forge Setup ==="

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/Scripts/activate 2>/dev/null || source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Verify GPU
echo ""
echo "=== GPU Verification ==="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
        print(f'GPU {i}: {name} ({mem:.1f} GB)')
    print('GPU setup OK!')
else:
    print('WARNING: No CUDA GPU detected. Training will be very slow on CPU.')
"

echo ""
echo "=== Setup Complete ==="
echo "Activate the environment with: source venv/Scripts/activate"
```

- [ ] **Step 3: Create setup.bat**

Create file `setup.bat`:

```bat
@echo off
echo === Narrative Forge Setup ===

if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

pip install --upgrade pip

echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo === GPU Verification ===
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('WARNING: No GPU detected')"

echo.
echo === Setup Complete ===
echo Activate with: venv\Scripts\activate.bat
```

- [ ] **Step 4: Wire up setup command in cli.py**

Replace the `main()` function in `cli.py`:

```python
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
        # Windows — check for bash (Git Bash) first, fall back to bat
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
```

- [ ] **Step 5: Run setup and verify**

```bash
cd C:\Users\nhf56\Documents\narrative-forge
python cli.py setup
```

Expected output: PyTorch version, CUDA available: True, both GPUs listed with 24.0 GB each.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt setup.sh setup.bat cli.py
git commit -m "feat: add Python environment setup with CUDA verification"
```

---

## Task 3: Configuration Files

**Files:**
- Create: `configs/training_config.yaml`
- Create: `configs/data_config.yaml`
- Create: `data/templates/default.yaml`

- [ ] **Step 1: Create training config**

Create file `configs/training_config.yaml`:

```yaml
# Model
base_model: "mistralai/Mistral-7B-Instruct-v0.3"
model_type: "causal_lm"

# QLoRA
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_use_double_quant: true

lora:
  r: 64
  lora_alpha: 128
  lora_dropout: 0.05
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  bias: "none"
  task_type: "CAUSAL_LM"

# Training
training:
  output_dir: "models/checkpoints"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 0.0002
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1
  max_seq_length: 2048
  fp16: true
  bf16: false
  logging_steps: 10
  save_strategy: "steps"
  save_steps: 500
  save_total_limit: 6
  optim: "paged_adamw_8bit"
  seed: 42

# Export
export:
  merged_dir: "models/merged"
  gguf_dir: "models/gguf"
  gguf_quantization: "q5_k_m"
  ollama_model_name: "narrative-forge"
```

- [ ] **Step 2: Create data config**

Create file `configs/data_config.yaml`:

```yaml
# Input
raw_dir: "data/raw"
supported_formats:
  - ".txt"
  - ".epub"
  - ".pdf"

# Chunking
chunking:
  min_chunk_words: 50
  max_chunk_words: 800
  overlap_sentences: 2

# Classification thresholds
classification:
  dialogue_quote_ratio: 0.3
  action_verb_ratio: 0.15
  internal_thought_indicators:
    - "thought"
    - "wondered"
    - "realized"
    - "felt"
    - "knew"
    - "considered"
    - "remembered"

# Output
processed_dir: "data/processed"
output_file: "training_data.jsonl"
stats_file: "stats.json"
train_split: 0.9
templates_file: "data/templates/default.yaml"
```

- [ ] **Step 3: Create prompt templates**

Create file `data/templates/default.yaml`:

```yaml
prose:
  - "Write a vivid, immersive description of {scene_hint}."
  - "Describe {scene_hint} with rich sensory detail and atmosphere."
  - "Paint a scene: {scene_hint}. Use evocative language."
  - "Write a passage that brings {scene_hint} to life."

dialogue:
  - "Write a dialogue scene between characters where {scene_hint}."
  - "Write a conversation that reveals character and advances tension: {scene_hint}."
  - "Craft a dialogue exchange: {scene_hint}."

action:
  - "Write an intense action sequence: {scene_hint}."
  - "Write a fast-paced scene of {scene_hint}."
  - "Describe {scene_hint} with visceral, kinetic prose."

worldbuilding:
  - "Describe the lore behind {scene_hint}, weaving it naturally into narrative."
  - "Explain {scene_hint} as it would appear in a fantasy novel — show, don't tell."
  - "Write a passage that reveals {scene_hint} through story, not exposition."

internal:
  - "Write a character's internal monologue as they {scene_hint}."
  - "Show a character's thoughts and emotions during {scene_hint}."
  - "Write an introspective passage: a character processes {scene_hint}."

transition:
  - "Write a scene transition: {scene_hint}."
  - "Bridge two scenes: {scene_hint}."
  - "Write an opening or closing passage for a chapter: {scene_hint}."
```

- [ ] **Step 4: Commit**

```bash
git add configs/ data/templates/
git commit -m "feat: add training, data, and prompt template configs"
```

---

## Task 4: Text Extraction Module

**Files:**
- Create: `src/extract.py`
- Create: `tests/conftest.py`
- Create: `tests/test_extract.py`

- [ ] **Step 1: Create test fixtures**

Create file `tests/conftest.py`:

```python
import os
import tempfile

import pytest


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_txt_file(tmp_dir):
    path = os.path.join(tmp_dir, "sample_book.txt")
    content = (
        "Chapter 1: The Beginning\n\n"
        "The forest stretched endlessly before her, ancient trees reaching "
        "toward a sky heavy with storm clouds. She could smell the rain coming, "
        "a clean sharp scent that cut through the decay of fallen leaves.\n\n"
        '"We should turn back," Marcus said, his hand on his sword hilt.\n\n'
        '"No." Sera kept walking. "The temple is close. I can feel it."\n\n'
        "The ground trembled beneath their feet. Somewhere deep below, "
        "something was waking.\n\n"
        "Chapter 2: The Temple\n\n"
        "They found it at dawn — a structure of black obsidian rising from "
        "the forest floor like a broken tooth. Equations were carved into "
        "every surface, mathematical truths that predated language itself.\n\n"
        "Sera pressed her palm against the stone. It was warm.\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


@pytest.fixture
def sample_raw_dir(tmp_dir, sample_txt_file):
    """A raw directory containing one sample book."""
    raw_dir = os.path.join(tmp_dir, "raw")
    os.makedirs(raw_dir)
    import shutil
    shutil.copy(sample_txt_file, os.path.join(raw_dir, "sample_book.txt"))
    return raw_dir
```

- [ ] **Step 2: Write failing tests for text extraction**

Create file `tests/test_extract.py`:

```python
import os

from src.extract import extract_text, extract_from_directory


class TestExtractText:
    def test_extracts_from_txt(self, sample_txt_file):
        result = extract_text(sample_txt_file)
        assert "forest stretched endlessly" in result
        assert "Sera pressed her palm" in result

    def test_strips_chapter_headers(self, sample_txt_file):
        result = extract_text(sample_txt_file)
        # Chapter headers are preserved as structural markers
        assert "Chapter 1" in result

    def test_returns_empty_for_missing_file(self, tmp_dir):
        result = extract_text(os.path.join(tmp_dir, "nonexistent.txt"))
        assert result == ""

    def test_handles_utf8_encoding(self, tmp_dir):
        path = os.path.join(tmp_dir, "unicode.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("The na\u00efve hero walked through the caf\u00e9.")
        result = extract_text(path)
        assert "na\u00efve" in result
        assert "caf\u00e9" in result


class TestExtractFromDirectory:
    def test_finds_all_txt_files(self, sample_raw_dir):
        results = extract_from_directory(sample_raw_dir)
        assert len(results) == 1
        assert results[0]["source"] == "sample_book.txt"
        assert "forest stretched endlessly" in results[0]["text"]

    def test_ignores_unsupported_formats(self, sample_raw_dir):
        # Add a .json file that should be ignored
        with open(os.path.join(sample_raw_dir, "notes.json"), "w") as f:
            f.write("{}")
        results = extract_from_directory(sample_raw_dir)
        assert len(results) == 1

    def test_empty_directory(self, tmp_dir):
        empty = os.path.join(tmp_dir, "empty")
        os.makedirs(empty)
        results = extract_from_directory(empty)
        assert results == []
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd C:\Users\nhf56\Documents\narrative-forge
venv/Scripts/activate && python -m pytest tests/test_extract.py -v
```

Expected: ImportError — `cannot import name 'extract_text' from 'src.extract'`

- [ ] **Step 4: Implement text extraction**

Create file `src/extract.py`:

```python
"""Extract clean text from book files (txt, epub, pdf)."""
import os
from pathlib import Path

SUPPORTED_FORMATS = {".txt", ".epub", ".pdf"}


def extract_text(file_path: str) -> str:
    """Extract text from a single file. Returns empty string on failure."""
    path = Path(file_path)

    if not path.exists():
        return ""

    ext = path.suffix.lower()

    if ext == ".txt":
        return _extract_txt(path)
    elif ext == ".epub":
        return _extract_epub(path)
    elif ext == ".pdf":
        return _extract_pdf(path)
    else:
        return ""


def extract_from_directory(dir_path: str) -> list[dict]:
    """Extract text from all supported files in a directory.

    Returns a list of dicts: [{"source": "filename.txt", "text": "..."}]
    """
    results = []
    dir_path = Path(dir_path)

    if not dir_path.exists():
        return results

    for file_path in sorted(dir_path.iterdir()):
        if file_path.suffix.lower() in SUPPORTED_FORMATS:
            text = extract_text(str(file_path))
            if text.strip():
                results.append({
                    "source": file_path.name,
                    "text": text,
                })

    return results


def _extract_txt(path: Path) -> str:
    """Read a plain text file."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return ""


def _extract_epub(path: Path) -> str:
    """Extract text from an EPUB file."""
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        print("WARNING: ebooklib/beautifulsoup4 not installed. Skipping EPUB.")
        return ""

    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    chapters = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n")
        text = text.strip()
        if text:
            chapters.append(text)

    return "\n\n".join(chapters)


def _extract_pdf(path: Path) -> str:
    """Extract text from a PDF file."""
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        print("WARNING: PyPDF2 not installed. Skipping PDF.")
        return ""

    reader = PdfReader(str(path))
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())

    return "\n\n".join(pages)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_extract.py -v
```

Expected: All 7 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/extract.py tests/conftest.py tests/test_extract.py
git commit -m "feat: add text extraction from txt, epub, and pdf files"
```

---

## Task 5: Text Chunking and Classification

**Files:**
- Create: `src/chunk.py`
- Create: `tests/test_chunk.py`

- [ ] **Step 1: Write failing tests**

Create file `tests/test_chunk.py`:

```python
from src.chunk import chunk_text, classify_chunk


class TestChunkText:
    def test_splits_on_paragraph_boundaries(self):
        text = (
            "First paragraph with enough words to form a complete thought "
            "about the world and its many wonders that stretch across the land.\n\n"
            "Second paragraph also with enough words to stand on its own "
            "as a meaningful piece of text in the narrative structure.\n\n"
            "Third paragraph completing the set with sufficient content "
            "to qualify as a proper chunk of narrative text here."
        )
        chunks = chunk_text(text, min_words=10, max_words=50)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk["text"].split()) >= 10

    def test_respects_max_words(self):
        text = " ".join(["word"] * 200)
        chunks = chunk_text(text, min_words=10, max_words=50)
        for chunk in chunks:
            assert len(chunk["text"].split()) <= 60  # allow some flexibility at boundaries

    def test_filters_short_chunks(self):
        text = "Too short.\n\nAlso too short.\n\n" + (
            "This paragraph is long enough to qualify as a real chunk of "
            "text with many words that tell a story about the world."
        )
        chunks = chunk_text(text, min_words=15, max_words=100)
        for chunk in chunks:
            assert len(chunk["text"].split()) >= 15


class TestClassifyChunk:
    def test_classifies_dialogue(self):
        text = (
            '"I will not go," she said firmly.\n'
            '"Then I will go alone," he replied.\n'
            '"You always say that," she whispered.'
        )
        assert classify_chunk(text) == "dialogue"

    def test_classifies_action(self):
        text = (
            "He swung his sword in a wide arc. The blade slashed through "
            "the air. She ducked, rolled, and struck back with her dagger. "
            "He blocked the blow and kicked her legs out from under her."
        )
        assert classify_chunk(text) == "action"

    def test_classifies_prose(self):
        text = (
            "The mountains rose in the distance, their peaks crowned with "
            "snow that gleamed in the fading light. The valley below was "
            "carpeted in wildflowers, purple and gold, swaying gently "
            "in the evening breeze."
        )
        assert classify_chunk(text) == "prose"

    def test_classifies_internal(self):
        text = (
            "She wondered if any of it had been real. The memories felt "
            "distant now, like dreams she could not quite hold. She realized "
            "that the person she had been before the war no longer existed."
        )
        assert classify_chunk(text) == "internal"

    def test_classifies_worldbuilding(self):
        text = (
            "The Binding was the fundamental law of all magic in Verathos. "
            "Every practitioner knew that power flowed from the gods through "
            "sacred bonds, and that each domain governed a specific aspect "
            "of reality. To violate a domain boundary was to court annihilation."
        )
        assert classify_chunk(text) == "worldbuilding"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_chunk.py -v
```

Expected: ImportError — `cannot import name 'chunk_text' from 'src.chunk'`

- [ ] **Step 3: Implement chunking and classification**

Create file `src/chunk.py`:

```python
"""Chunk text into passages and classify by narrative type."""
import re

# Action verbs commonly found in combat/chase/physical sequences
ACTION_VERBS = {
    "swung", "slashed", "struck", "blocked", "kicked", "punched", "dodged",
    "ducked", "rolled", "charged", "leaped", "sprinted", "crashed", "smashed",
    "thrust", "parried", "stabbed", "hurled", "lunged", "tackled", "slammed",
    "dove", "fired", "shot", "threw", "caught", "grabbed", "shoved", "pulled",
    "ran", "fled", "chased",
}

# Words indicating internal thought/reflection
THOUGHT_INDICATORS = {
    "thought", "wondered", "realized", "felt", "knew", "considered",
    "remembered", "imagined", "wished", "hoped", "feared", "believed",
    "pondered", "reflected", "recalled", "supposed", "doubted",
}

# Words indicating world-building/lore exposition
LORE_INDICATORS = {
    "law", "laws", "magic", "power", "domain", "domains", "ancient",
    "tradition", "ritual", "sacred", "forbidden", "practiced", "governed",
    "fundamental", "practitioner", "practitioners", "bond", "bonds",
    "rule", "rules", "system", "order", "hierarchy",
}


def chunk_text(text: str, min_words: int = 50, max_words: int = 800) -> list[dict]:
    """Split text into chunks on paragraph boundaries.

    Returns list of dicts: [{"text": "...", "word_count": N}]
    """
    # Split on double newlines (paragraph boundaries)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    current_chunk = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        # If adding this paragraph exceeds max, flush current chunk
        if current_words + para_words > max_words and current_chunk:
            chunk_text_joined = "\n\n".join(current_chunk)
            if current_words >= min_words:
                chunks.append({
                    "text": chunk_text_joined,
                    "word_count": current_words,
                })
            current_chunk = []
            current_words = 0

        current_chunk.append(para)
        current_words += para_words

    # Flush remaining
    if current_chunk and current_words >= min_words:
        chunks.append({
            "text": "\n\n".join(current_chunk),
            "word_count": current_words,
        })

    return chunks


def classify_chunk(text: str) -> str:
    """Classify a text chunk into a narrative type.

    Returns one of: dialogue, action, internal, worldbuilding, transition, prose
    """
    words = text.lower().split()
    word_count = len(words)
    if word_count == 0:
        return "prose"

    word_set = set(words)

    # Count dialogue indicators (quoted speech)
    quote_lines = len(re.findall(r'["\u201c].+?["\u201d]', text))
    total_lines = max(len(text.strip().split("\n")), 1)
    dialogue_ratio = quote_lines / total_lines

    if dialogue_ratio >= 0.3:
        return "dialogue"

    # Count action verbs
    action_count = len(word_set & ACTION_VERBS)
    action_ratio = action_count / max(word_count, 1)

    if action_ratio >= 0.02 and action_count >= 3:
        return "action"

    # Count thought indicators
    thought_count = len(word_set & THOUGHT_INDICATORS)

    if thought_count >= 3:
        return "internal"

    # Count lore/worldbuilding indicators
    lore_count = len(word_set & LORE_INDICATORS)

    if lore_count >= 3:
        return "worldbuilding"

    # Check for transition patterns
    transition_patterns = [
        r"^(later|afterward|the next|days passed|time passed|when \w+ woke)",
        r"(chapter \d+|part \w+)",
    ]
    for pattern in transition_patterns:
        if re.search(pattern, text.lower()):
            return "transition"

    # Default: prose/description
    return "prose"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_chunk.py -v
```

Expected: All 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/chunk.py tests/test_chunk.py
git commit -m "feat: add text chunking and narrative classification"
```

---

## Task 6: Training Pair Generator

**Files:**
- Create: `src/pair_generator.py`
- Create: `tests/test_pair_generator.py`

- [ ] **Step 1: Write failing tests**

Create file `tests/test_pair_generator.py`:

```python
import os
import yaml

from src.pair_generator import generate_pairs, generate_scene_hint


class TestGenerateSceneHint:
    def test_produces_nonempty_hint(self):
        text = (
            "The mountains rose in the distance, their peaks crowned with "
            "snow that gleamed in the fading light."
        )
        hint = generate_scene_hint(text)
        assert isinstance(hint, str)
        assert len(hint) > 10
        assert len(hint) < 200

    def test_different_texts_produce_different_hints(self):
        text1 = "A dark forest filled with ancient trees and hidden dangers."
        text2 = "A bustling city market overflowing with spices and silk."
        hint1 = generate_scene_hint(text1)
        hint2 = generate_scene_hint(text2)
        assert hint1 != hint2


class TestGeneratePairs:
    def test_produces_instruction_completion_pairs(self):
        chunks = [
            {"text": "The forest was dark and old.", "type": "prose"},
        ]
        templates = {
            "prose": ["Write a description of {scene_hint}."],
        }
        pairs = generate_pairs(chunks, templates)
        assert len(pairs) == 1
        assert "instruction" in pairs[0]
        assert "completion" in pairs[0]
        assert "type" in pairs[0]
        assert pairs[0]["completion"] == "The forest was dark and old."
        assert pairs[0]["type"] == "prose"

    def test_uses_correct_template_for_type(self):
        chunks = [
            {"text": '"Hello," she said.', "type": "dialogue"},
        ]
        templates = {
            "dialogue": ["Write dialogue: {scene_hint}."],
            "prose": ["Write prose: {scene_hint}."],
        }
        pairs = generate_pairs(chunks, templates)
        assert pairs[0]["instruction"].startswith("Write dialogue:")

    def test_handles_missing_template_type(self):
        chunks = [
            {"text": "Some text here.", "type": "unknown_type"},
        ]
        templates = {
            "prose": ["Write: {scene_hint}."],
        }
        pairs = generate_pairs(chunks, templates)
        # Falls back to prose template
        assert len(pairs) == 1

    def test_multiple_chunks(self):
        chunks = [
            {"text": "Forest description.", "type": "prose"},
            {"text": '"Hello," she said.', "type": "dialogue"},
            {"text": "He swung his sword.", "type": "action"},
        ]
        templates = {
            "prose": ["Describe: {scene_hint}."],
            "dialogue": ["Write dialogue: {scene_hint}."],
            "action": ["Write action: {scene_hint}."],
        }
        pairs = generate_pairs(chunks, templates)
        assert len(pairs) == 3
        types = [p["type"] for p in pairs]
        assert "prose" in types
        assert "dialogue" in types
        assert "action" in types
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_pair_generator.py -v
```

Expected: ImportError — `cannot import name 'generate_pairs' from 'src.pair_generator'`

- [ ] **Step 3: Implement pair generator**

Create file `src/pair_generator.py`:

```python
"""Generate instruction/completion training pairs from classified chunks."""
import random
import re


def generate_scene_hint(text: str) -> str:
    """Extract a short scene hint from a text chunk.

    Takes the first 1-2 sentences and strips them down to a concise description
    that can fill the {scene_hint} slot in a template.
    """
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    # Take first 1-2 sentences
    hint_sentences = sentences[:2]
    hint = " ".join(hint_sentences)

    # Truncate to ~150 chars if needed
    if len(hint) > 150:
        hint = hint[:147] + "..."

    # Clean up
    hint = hint.strip().rstrip(".")
    hint = hint[0].lower() + hint[1:] if hint else hint

    return hint


def generate_pairs(
    chunks: list[dict], templates: dict[str, list[str]]
) -> list[dict]:
    """Generate instruction/completion pairs from classified chunks.

    Args:
        chunks: List of {"text": str, "type": str}
        templates: Dict mapping chunk type to list of instruction templates.
                   Templates use {scene_hint} placeholder.

    Returns:
        List of {"instruction": str, "completion": str, "type": str}
    """
    pairs = []
    fallback_type = "prose"

    for chunk in chunks:
        chunk_type = chunk["type"]
        text = chunk["text"]

        # Get templates for this type, fall back to prose
        type_templates = templates.get(chunk_type)
        if not type_templates:
            type_templates = templates.get(fallback_type, [])
        if not type_templates:
            continue

        # Pick a random template
        template = random.choice(type_templates)

        # Generate scene hint from the text
        hint = generate_scene_hint(text)

        # Fill template
        instruction = template.replace("{scene_hint}", hint)

        pairs.append({
            "instruction": instruction,
            "completion": text,
            "type": chunk_type,
        })

    return pairs
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_pair_generator.py -v
```

Expected: All 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pair_generator.py tests/test_pair_generator.py
git commit -m "feat: add instruction/completion pair generator"
```

---

## Task 7: Data Preparation Pipeline (Orchestrator)

**Files:**
- Create: `src/prepare.py`
- Create: `tests/test_prepare.py`
- Modify: `cli.py` — wire up `prepare` command

- [ ] **Step 1: Write failing tests**

Create file `tests/test_prepare.py`:

```python
import json
import os

from src.prepare import run_prepare


class TestRunPrepare:
    def test_produces_jsonl_output(self, sample_raw_dir, tmp_dir):
        output_dir = os.path.join(tmp_dir, "processed")
        os.makedirs(output_dir)

        templates = {
            "prose": ["Write: {scene_hint}."],
            "dialogue": ["Write dialogue: {scene_hint}."],
            "action": ["Write action: {scene_hint}."],
            "internal": ["Write thoughts: {scene_hint}."],
            "worldbuilding": ["Write lore: {scene_hint}."],
            "transition": ["Write transition: {scene_hint}."],
        }

        stats = run_prepare(
            raw_dir=sample_raw_dir,
            output_dir=output_dir,
            templates=templates,
            min_chunk_words=10,
            max_chunk_words=200,
        )

        # Check JSONL file was created
        train_path = os.path.join(output_dir, "train.jsonl")
        assert os.path.exists(train_path)

        # Check each line is valid JSON with required fields
        with open(train_path) as f:
            for line in f:
                record = json.loads(line)
                assert "instruction" in record
                assert "completion" in record
                assert "type" in record

        # Check stats
        assert stats["total_pairs"] > 0
        assert stats["total_books"] == 1
        assert "type_breakdown" in stats

    def test_creates_train_and_val_splits(self, sample_raw_dir, tmp_dir):
        output_dir = os.path.join(tmp_dir, "processed")
        os.makedirs(output_dir)

        templates = {
            "prose": ["Write: {scene_hint}."],
            "dialogue": ["Write dialogue: {scene_hint}."],
            "action": ["Write action: {scene_hint}."],
            "internal": ["Write thoughts: {scene_hint}."],
            "worldbuilding": ["Write lore: {scene_hint}."],
            "transition": ["Write transition: {scene_hint}."],
        }

        run_prepare(
            raw_dir=sample_raw_dir,
            output_dir=output_dir,
            templates=templates,
            min_chunk_words=10,
            max_chunk_words=200,
        )

        train_path = os.path.join(output_dir, "train.jsonl")
        val_path = os.path.join(output_dir, "val.jsonl")
        assert os.path.exists(train_path)
        assert os.path.exists(val_path)

    def test_writes_stats_file(self, sample_raw_dir, tmp_dir):
        output_dir = os.path.join(tmp_dir, "processed")
        os.makedirs(output_dir)

        templates = {
            "prose": ["Write: {scene_hint}."],
            "dialogue": ["Write dialogue: {scene_hint}."],
            "action": ["Write action: {scene_hint}."],
            "internal": ["Write thoughts: {scene_hint}."],
            "worldbuilding": ["Write lore: {scene_hint}."],
            "transition": ["Write transition: {scene_hint}."],
        }

        run_prepare(
            raw_dir=sample_raw_dir,
            output_dir=output_dir,
            templates=templates,
            min_chunk_words=10,
            max_chunk_words=200,
        )

        stats_path = os.path.join(output_dir, "stats.json")
        assert os.path.exists(stats_path)

        with open(stats_path) as f:
            stats = json.load(f)
        assert "total_pairs" in stats
        assert "total_books" in stats
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_prepare.py -v
```

Expected: ImportError — `cannot import name 'run_prepare' from 'src.prepare'`

- [ ] **Step 3: Implement prepare pipeline**

Create file `src/prepare.py`:

```python
"""Orchestrate the full data preparation pipeline: extract -> chunk -> pair -> JSONL."""
import json
import os
import random
from collections import Counter

import yaml

from src.extract import extract_from_directory
from src.chunk import chunk_text, classify_chunk
from src.pair_generator import generate_pairs


def run_prepare(
    raw_dir: str,
    output_dir: str,
    templates: dict[str, list[str]],
    min_chunk_words: int = 50,
    max_chunk_words: int = 800,
    train_split: float = 0.9,
    seed: int = 42,
) -> dict:
    """Run the full data preparation pipeline.

    Args:
        raw_dir: Directory containing raw book files
        output_dir: Directory to write JSONL output
        templates: Instruction templates per chunk type
        min_chunk_words: Minimum words per chunk
        max_chunk_words: Maximum words per chunk
        train_split: Fraction of data for training (rest is validation)
        seed: Random seed for reproducibility

    Returns:
        Stats dict with counts and breakdowns
    """
    random.seed(seed)

    # Step 1: Extract text from all books
    print(f"Extracting text from {raw_dir}...")
    books = extract_from_directory(raw_dir)
    print(f"  Found {len(books)} book(s)")

    if not books:
        print("  WARNING: No books found. Add files to data/raw/")
        return {"total_pairs": 0, "total_books": 0, "type_breakdown": {}}

    # Step 2: Chunk and classify
    all_chunks = []
    for book in books:
        chunks = chunk_text(book["text"], min_chunk_words, max_chunk_words)
        for chunk in chunks:
            chunk["type"] = classify_chunk(chunk["text"])
            chunk["source"] = book["source"]
        all_chunks.extend(chunks)
        print(f"  {book['source']}: {len(chunks)} chunks")

    print(f"  Total chunks: {len(all_chunks)}")

    # Step 3: Generate instruction/completion pairs
    pairs = generate_pairs(all_chunks, templates)
    print(f"  Generated {len(pairs)} training pairs")

    # Step 4: Shuffle and split
    random.shuffle(pairs)
    split_idx = int(len(pairs) * train_split)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    # Step 5: Write JSONL files
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.jsonl")
    _write_jsonl(train_pairs, train_path)
    print(f"  Train set: {len(train_pairs)} pairs -> {train_path}")

    val_path = os.path.join(output_dir, "val.jsonl")
    _write_jsonl(val_pairs, val_path)
    print(f"  Val set: {len(val_pairs)} pairs -> {val_path}")

    # Step 6: Compute and write stats
    type_counts = Counter(p["type"] for p in pairs)
    source_counts = Counter(c["source"] for c in all_chunks)
    avg_completion_len = (
        sum(len(p["completion"].split()) for p in pairs) / max(len(pairs), 1)
    )

    stats = {
        "total_pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "total_books": len(books),
        "type_breakdown": dict(type_counts),
        "source_breakdown": dict(source_counts),
        "avg_completion_words": round(avg_completion_len, 1),
    }

    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats written to {stats_path}")

    return stats


def load_templates(templates_path: str) -> dict[str, list[str]]:
    """Load instruction templates from a YAML file."""
    with open(templates_path) as f:
        return yaml.safe_load(f)


def _write_jsonl(records: list[dict], path: str):
    """Write records as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_prepare.py -v
```

Expected: All 3 tests pass.

- [ ] **Step 5: Wire up prepare command in cli.py**

Add to `cli.py` — add this function and update the `main()` dispatch:

```python
def run_prepare_cmd():
    """Run the data preparation pipeline."""
    import yaml
    # Ensure we can import src
    sys.path.insert(0, os.path.dirname(__file__))
    from src.prepare import run_prepare, load_templates

    # Load data config
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
```

Update the dispatch in `main()`:

```python
    if command == "setup":
        run_setup()
    elif command == "prepare":
        run_prepare_cmd()
    else:
        print(f"Command '{command}' not yet implemented.")
        sys.exit(1)
```

- [ ] **Step 6: Commit**

```bash
git add src/prepare.py tests/test_prepare.py cli.py
git commit -m "feat: add data preparation pipeline with train/val split"
```

---

## Task 8: QLoRA Training Script

**Files:**
- Create: `src/train.py`
- Modify: `cli.py` — wire up `train` command

- [ ] **Step 1: Implement training script**

Create file `src/train.py`:

```python
"""QLoRA fine-tuning script for narrative generation."""
import os

import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def format_training_example(example: dict, tokenizer) -> str:
    """Format an instruction/completion pair for training.

    Uses the chat template format that Mistral expects.
    """
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def run_training(config_path: str = "configs/training_config.yaml"):
    """Run QLoRA fine-tuning."""
    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("=== Narrative Forge — QLoRA Training ===\n")

    model_name = cfg["base_model"]
    print(f"Base model: {model_name}")

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available. Training requires a GPU.")
        return
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB\n")

    # Quantization config
    quant_cfg = cfg["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model in 4-bit
    print("Loading model in 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
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

    # Load dataset
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

    # Training arguments
    train_cfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        optim=train_cfg["optim"],
        seed=train_cfg["seed"],
        report_to="none",
        remove_unused_columns=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
        formatting_func=lambda ex: format_training_example(ex, tokenizer),
        max_seq_length=train_cfg["max_seq_length"],
    )

    # Train
    print("Starting training...\n")
    trainer.train()

    # Save final adapter
    final_adapter_dir = os.path.join(train_cfg["output_dir"], "final_adapter")
    trainer.save_model(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print(f"\nTraining complete! Adapter saved to {final_adapter_dir}")
    print("Next step: python cli.py merge")
```

- [ ] **Step 2: Wire up train command in cli.py**

Add to `cli.py`:

```python
def run_train_cmd():
    """Run QLoRA fine-tuning."""
    sys.path.insert(0, os.path.dirname(__file__))
    from src.train import run_training
    run_training()
```

Update dispatch in `main()`:

```python
    if command == "setup":
        run_setup()
    elif command == "prepare":
        run_prepare_cmd()
    elif command == "train":
        run_train_cmd()
    else:
        print(f"Command '{command}' not yet implemented.")
        sys.exit(1)
```

- [ ] **Step 3: Commit**

```bash
git add src/train.py cli.py
git commit -m "feat: add QLoRA training script with config-driven hyperparameters"
```

---

## Task 9: Evaluation Script

**Files:**
- Create: `src/evaluate.py`
- Modify: `cli.py` — wire up `evaluate` command

- [ ] **Step 1: Implement evaluation script**

Create file `src/evaluate.py`:

```python
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

    # Load model + adapter
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

    # Generate samples
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

    # Write samples to file
    output_path = os.path.join(output_dir, f"eval_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    # Also write a readable markdown version
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
```

- [ ] **Step 2: Wire up evaluate command in cli.py**

Add to `cli.py`:

```python
def run_evaluate_cmd():
    """Generate sample outputs for review."""
    sys.path.insert(0, os.path.dirname(__file__))
    from src.evaluate import run_evaluation
    run_evaluation()
```

Update dispatch in `main()`:

```python
    if command == "setup":
        run_setup()
    elif command == "prepare":
        run_prepare_cmd()
    elif command == "train":
        run_train_cmd()
    elif command == "evaluate":
        run_evaluate_cmd()
    else:
        print(f"Command '{command}' not yet implemented.")
        sys.exit(1)
```

- [ ] **Step 3: Commit**

```bash
git add src/evaluate.py cli.py
git commit -m "feat: add evaluation script with sample generation and markdown output"
```

---

## Task 10: Adapter Merge Script

**Files:**
- Create: `src/merge.py`
- Modify: `cli.py` — wire up `merge` command

- [ ] **Step 1: Implement merge script**

Create file `src/merge.py`:

```python
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

    # Load tokenizer
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load base model in full precision for merging
    print(f"Loading base model in FP16 (this uses ~14GB RAM)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",  # merge on CPU to avoid VRAM limits
    )

    # Load and merge adapter
    print(f"Loading adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(model, adapter_dir)

    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    # Save merged model
    os.makedirs(merged_dir, exist_ok=True)
    print(f"Saving merged model to {merged_dir}...")
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    print(f"\nMerge complete! Merged model saved to {merged_dir}")
    print("Next step: python cli.py export")
```

- [ ] **Step 2: Wire up merge command in cli.py**

Add to `cli.py`:

```python
def run_merge_cmd():
    """Merge LoRA adapter into base model."""
    sys.path.insert(0, os.path.dirname(__file__))
    from src.merge import run_merge
    run_merge()
```

Update dispatch in `main()`:

```python
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
    else:
        print(f"Command '{command}' not yet implemented.")
        sys.exit(1)
```

- [ ] **Step 3: Commit**

```bash
git add src/merge.py cli.py
git commit -m "feat: add LoRA adapter merge script"
```

---

## Task 11: GGUF Export Script

**Files:**
- Create: `src/export.py`
- Modify: `cli.py` — wire up `export` command

- [ ] **Step 1: Implement export script**

Create file `src/export.py`:

```python
"""Convert merged model to GGUF format for Ollama."""
import os
import subprocess
import sys

import yaml


def run_export(config_path: str = "configs/training_config.yaml"):
    """Convert the merged model to GGUF format using llama.cpp."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    merged_dir = cfg["export"]["merged_dir"]
    gguf_dir = cfg["export"]["gguf_dir"]
    quantization = cfg["export"]["gguf_quantization"]
    model_name = cfg["export"]["ollama_model_name"]

    if not os.path.exists(merged_dir):
        print(f"ERROR: No merged model found at {merged_dir}")
        print("Run 'python cli.py merge' first.")
        return

    print("=== Narrative Forge — Export to GGUF ===\n")

    os.makedirs(gguf_dir, exist_ok=True)

    # Use llama-cpp-python's conversion or huggingface_hub's GGUF support
    output_path = os.path.join(gguf_dir, f"{model_name}.gguf")

    # Method: Use the huggingface-hub CLI to convert
    # First check if llama.cpp convert script is available
    llama_cpp_convert = _find_llama_cpp_convert()

    if llama_cpp_convert:
        print("Using llama.cpp for conversion...")
        _convert_with_llama_cpp(llama_cpp_convert, merged_dir, gguf_dir, model_name, quantization)
    else:
        print("Using llama-cpp-python for conversion...")
        _convert_with_python(merged_dir, output_path, quantization)

    if os.path.exists(output_path):
        size_gb = os.path.getsize(output_path) / 1024**3
        print(f"\nExport complete! GGUF model saved to {output_path} ({size_gb:.1f} GB)")
        print("Next step: python cli.py register")
    else:
        print("\nERROR: GGUF file was not created. Check the output above for errors.")


def _find_llama_cpp_convert():
    """Check if llama.cpp convert script is available."""
    try:
        result = subprocess.run(
            ["python", "-c", "import llama_cpp; print(llama_cpp.__file__)"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return "llama_cpp"
    except FileNotFoundError:
        pass
    return None


def _convert_with_python(merged_dir: str, output_path: str, quantization: str):
    """Convert using the transformers/huggingface_hub GGUF export."""
    try:
        # Use the convert-to-gguf approach via llama.cpp's python script
        # This is bundled with llama-cpp-python
        subprocess.run(
            [
                sys.executable, "-m", "llama_cpp.llama_cpp",
                "--convert", merged_dir,
                "--outfile", output_path,
                "--outtype", quantization,
            ],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: clone llama.cpp and use convert_hf_to_gguf.py
        print("Installing llama.cpp conversion tools...")
        llama_cpp_dir = os.path.join("models", "llama.cpp")
        if not os.path.exists(llama_cpp_dir):
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/ggerganov/llama.cpp.git", llama_cpp_dir],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r",
                 os.path.join(llama_cpp_dir, "requirements.txt")],
                check=True,
            )

        # Convert to f16 GGUF first
        f16_path = output_path.replace(".gguf", "-f16.gguf")
        subprocess.run(
            [sys.executable, os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py"),
             merged_dir, "--outfile", f16_path, "--outtype", "f16"],
            check=True,
        )

        # Then quantize
        quantize_bin = os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize")
        if os.path.exists(quantize_bin):
            subprocess.run([quantize_bin, f16_path, output_path, quantization], check=True)
        else:
            # If no compiled binary, just use the f16 version
            os.rename(f16_path, output_path)
            print(f"NOTE: Could not quantize to {quantization}. Using f16 instead.")
            print("For quantization, build llama.cpp: cd models/llama.cpp && cmake -B build && cmake --build build")


def _convert_with_llama_cpp(converter, merged_dir, gguf_dir, model_name, quantization):
    """Convert using llama.cpp tools."""
    output_path = os.path.join(gguf_dir, f"{model_name}.gguf")
    _convert_with_python(merged_dir, output_path, quantization)
```

- [ ] **Step 2: Wire up export command in cli.py**

Add to `cli.py`:

```python
def run_export_cmd():
    """Convert merged model to GGUF."""
    sys.path.insert(0, os.path.dirname(__file__))
    from src.export import run_export
    run_export()
```

Update dispatch in `main()`:

```python
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
    else:
        print(f"Command '{command}' not yet implemented.")
        sys.exit(1)
```

- [ ] **Step 3: Commit**

```bash
git add src/export.py cli.py
git commit -m "feat: add GGUF export with llama.cpp conversion"
```

---

## Task 12: Ollama Registration

**Files:**
- Create: `Modelfile`
- Modify: `cli.py` — wire up `register` command

- [ ] **Step 1: Create Modelfile template**

Create file `Modelfile`:

```
FROM ./models/gguf/narrative-forge.gguf

TEMPLATE """{{- if .System }}{{ .System }}
{{ end }}{{ if .Prompt }}{{ .Prompt }}
{{ end }}{{ .Response }}"""

SYSTEM """You are Narrative Forge, a creative writing model fine-tuned on masterful fantasy fiction. You write vivid prose, compelling dialogue, atmospheric descriptions, and well-paced narrative. You absorb creative briefs that describe what a scene should accomplish — characters involved, emotional beats, constraints, tone — and produce polished prose that brings them to life."""

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048
```

- [ ] **Step 2: Add register logic to cli.py**

Add to `cli.py`:

```python
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

    # Check Ollama is installed
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: Ollama is not installed.")
        print("Install from: https://ollama.com/download")
        print("After installing, run this command again.")
        sys.exit(1)

    print(f"=== Registering {model_name} with Ollama ===\n")

    # Create model from Modelfile
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
```

Update dispatch in `main()`:

```python
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
```

- [ ] **Step 3: Commit**

```bash
git add Modelfile cli.py
git commit -m "feat: add Ollama model registration with Modelfile"
```

---

## Task 13: Install Ollama

**Files:** None — system install

- [ ] **Step 1: Download and install Ollama**

Download from https://ollama.com/download and install. On Windows this is a standard installer.

- [ ] **Step 2: Verify installation**

```bash
ollama --version
ollama list
```

Expected: Version number printed, empty model list.

- [ ] **Step 3: Test Ollama is serving**

```bash
curl http://localhost:11434/api/tags
```

Expected: JSON response with empty models list `{"models":[]}`.

---

## Task 14: End-to-End Smoke Test

**Files:** None — manual verification

- [ ] **Step 1: Place a test book in data/raw/**

Use a public domain text. Download a short Project Gutenberg fantasy text:

```bash
cd C:\Users\nhf56\Documents\narrative-forge
curl -o data/raw/grimm_fairy_tales.txt https://www.gutenberg.org/cache/epub/2591/pg2591.txt
```

- [ ] **Step 2: Run the full pipeline**

```bash
# Activate venv
source venv/Scripts/activate

# Prepare data
python cli.py prepare

# Verify output
cat data/processed/stats.json
head -5 data/processed/train.jsonl
```

Expected: Stats showing multiple training pairs across different types.

- [ ] **Step 3: Run training (short test — 1 epoch)**

Edit `configs/training_config.yaml` temporarily: set `num_train_epochs: 1` and `save_steps: 100`.

```bash
python cli.py train
```

Expected: Training starts, loss decreases over steps, adapter saved to `models/checkpoints/final_adapter/`.

- [ ] **Step 4: Evaluate**

```bash
python cli.py evaluate
```

Expected: Sample outputs generated in `models/samples/`, readable in the markdown file.

- [ ] **Step 5: Full export pipeline**

```bash
python cli.py merge
python cli.py export
python cli.py register
ollama run narrative-forge "Describe an ancient temple at dawn."
```

Expected: Model responds with generated prose.

- [ ] **Step 6: Restore training config and commit**

Restore `num_train_epochs: 3` and `save_steps: 500` in the config.

```bash
git add -A
git commit -m "test: verify end-to-end pipeline with public domain text"
git push
```
