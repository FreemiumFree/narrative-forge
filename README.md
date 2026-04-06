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
