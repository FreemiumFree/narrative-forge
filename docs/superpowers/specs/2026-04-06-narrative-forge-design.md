# Narrative Forge — Design Spec

**Date**: 2026-04-06
**Status**: Draft
**Related project**: fantasy-book-engine (`C:\Users\nhf56\Documents\fantasy-book-engine`)

---

## Purpose

Narrative Forge is an LLM fine-tuning project that trains an open-source language model on the user's favorite fantasy novels (both owned and public domain) to produce a local, offline creative writing engine. The fine-tuned model absorbs prose style, narrative structure, world-building voice, dialogue patterns, and plot pacing from the training corpus.

It works in tandem with the fantasy-book-engine project:
- **narrative-forge** generates all creative content — prose, plot beats, scene pacing, dialogue, conflict design — powered by a model trained on books the user loves.
- **fantasy-book-engine** orchestrates the process — it holds the world bible (PostgreSQL + pgvector), gathers context for each creative request, validates generated content against world rules and continuity, and commits approved content as series canon.

The two projects communicate over HTTP. Fantasy-book-engine sends structured creative briefs to narrative-forge's model (served via Ollama), receives generated content, validates it, and either commits it or sends it back with revision feedback.

## Hardware

| Component | Specs |
|-----------|-------|
| GPU | 2x NVIDIA TITAN RTX (24GB VRAM each, 48GB total, Turing arch / compute 7.5) |
| CPU | Intel Core i9-7980XE @ 2.60GHz (18 cores / 36 threads) |
| RAM | 128GB (8x 16GB) |
| Storage | ~15TB SSD (Samsung 860 EVO 1TB + 870 QVO 8TB + 970 EVO Plus 2TB + 860 EVO 4TB) |

**Hardware constraints**:
- Turing GPUs support FP16 but not BF16 natively — all training uses FP16
- 48GB VRAM supports full fine-tune of 7B models or QLoRA of up to 13B

## Approach

**QLoRA fine-tuning** — trains small adapter layers (~1-4% of parameters) on top of a frozen 4-bit quantized base model.

Rationale:
- Fast iteration (~30-60 min per epoch) for rapid experimentation with data mixes and hyperparameters
- Fits comfortably on a single TITAN RTX, keeping the second GPU free for inference/testing
- Quality gap vs. full fine-tune is marginal for creative writing / style transfer
- Adapters can be merged into the base model and exported to Ollama when satisfied
- Path to scale: can layer continual pre-training underneath later without starting over

**Model progression**: Start with 7B, scale to 13B once the pipeline and data are dialed in.

## Base Model

**Mistral 7B Instruct v0.3**

- Best-in-class 7B for creative writing with strong instruction following
- Apache 2.0 license (free for any use)
- Well-supported across the Hugging Face ecosystem
- Upgradeable: same pipeline works for Mistral 13B or Llama 3 models later
- Note: If a newer Mistral version is available at implementation time, prefer the latest stable instruct variant

## Project Structure

```
narrative-forge/
├── data/
│   ├── raw/                  # Original book text files (gitignored)
│   ├── processed/            # Cleaned, chunked training data
│   └── templates/            # Prompt templates for structuring training pairs
├── scripts/
│   ├── prepare_data.py       # Book text -> structured training pairs
│   ├── train.py              # QLoRA fine-tuning script
│   ├── merge_adapter.py      # Merge LoRA weights into base model
│   ├── export_ollama.py      # Convert merged model -> GGUF -> Ollama
│   └── evaluate.py           # Generate sample outputs for manual review
├── configs/
│   ├── training_config.yaml  # Hyperparameters, model selection, paths
│   └── data_config.yaml      # Data processing settings
├── models/                   # Output models & adapters (gitignored)
├── notebooks/                # Jupyter notebooks for exploration
├── requirements.txt          # Python dependencies
├── setup.sh                  # One-command environment setup
├── Makefile                  # Simple commands: make prepare, make train, etc.
├── .gitignore
└── README.md
```

**Gitignored**: `data/raw/` (copyright), `models/` (too large), `.env`, `__pycache__/`, `*.pyc`

## Data Pipeline

### Phase 1: Text Extraction
- Input: book files in `data/raw/` (`.txt`, `.epub`, `.pdf`)
- `prepare_data.py` extracts clean text, strips headers/footers/page numbers, normalizes formatting

### Phase 2: Chunking & Classification
Segments each book into labeled chunks by narrative type:
- **Prose/Description** — landscape, atmosphere, scene-setting
- **Dialogue** — character conversations with surrounding context
- **Action** — combat, chase, physical sequences
- **World-building** — lore exposition, magic system explanations, cultural descriptions
- **Internal monologue** — character thoughts, emotional beats
- **Transition** — chapter openings/closings, time skips, scene shifts

Classification uses heuristics (dialogue punctuation, action verbs, paragraph patterns). Manual correction supported for higher quality.

### Phase 3: Training Pair Generation
Each chunk is wrapped into an instruction/completion pair using configurable templates from `data/templates/`.

Example pairs:
```
Instruction: "Write a vivid description of an ancient forest that feels sacred and dangerous."
Completion:  [extracted prose passage about a forest]

Instruction: "Write a tense dialogue between a mentor and a reluctant hero who has just discovered their power."
Completion:  [extracted dialogue scene]

Instruction: "Describe the magic system of a world where power flows from divine bonds."
Completion:  [extracted world-building passage]
```

### Phase 4: Dataset Output
- Format: JSONL (`data/processed/training_data.jsonl`)
- Split: 90% train / 10% validation
- Stats logged to `data/processed/stats.json`: total pairs, breakdown by type, average length, source book distribution

## Training Pipeline

### QLoRA Configuration
- **Quantization**: 4-bit NormalFloat (NF4)
- **LoRA rank**: 64, alpha: 128
- **Target modules**: All attention layers (q, k, v, o) + MLP layers
- **Precision**: FP16 (Turing GPU constraint)
- **GPU**: Single TITAN RTX (second GPU free for testing)

### Hyperparameters
- Batch size: 4, gradient accumulation steps: 8 (effective batch 32)
- Learning rate: 2e-4, cosine scheduler
- Epochs: 3 with checkpoints every 0.5 epoch
- Max sequence length: 2048 tokens
- Warmup: 10% of total steps

### Dependencies
- `transformers` — model loading and training
- `peft` — QLoRA/LoRA implementation
- `bitsandbytes` — 4-bit quantization
- `datasets` — data loading
- `accelerate` — GPU management
- `trl` — SFTTrainer for supervised fine-tuning

### Estimated Training Time
~30-60 minutes per epoch for a typical corpus (50k-200k training pairs). Full 3-epoch run: ~1.5-3 hours.

## Model Export & Serving

### Step 1: Merge Adapter (`make merge`)
Merges LoRA adapter weights into the base model. Output: full standalone model in `models/merged/`.

### Step 2: Convert to GGUF (`make export`)
Converts merged model to GGUF format with Q5_K_M quantization (quality/speed balance). Output: `models/gguf/narrative-forge.gguf`.

### Step 3: Register with Ollama (`make register`)
Creates an Ollama Modelfile with system prompt and parameters. Registers as `narrative-forge` in Ollama. Accessible via `ollama run narrative-forge` or `http://localhost:11434/api/generate`.

## Integration with Fantasy Book Engine

### Architecture

```
fantasy-book-engine (guardian)          narrative-forge (creative writer)
┌──────────────────────────┐           ┌──────────────────────────┐
│ Orchestrator gathers     │  brief    │ Fine-tuned model         │
│ context from world bible ├──────────>│ generates creative       │
│ (characters, divine      │           │ content using learned    │
│ state, plot position,    │  content  │ style, pacing, voice     │
│ open threads, reader     │<──────────┤ from training corpus     │
│ knowledge)               │           │                          │
│                          │           │ Served via Ollama at     │
│ QC agents validate       │  revision │ localhost:11434          │
│ against bible, commit    ├──────────>│                          │
│ or send back for         │           │                          │
│ revision                 │           │                          │
└──────────────────────────┘           └──────────────────────────┘
```

### Communication Protocol
HTTP calls to Ollama API at `http://localhost:11434/api/generate`:

```json
{
  "model": "narrative-forge",
  "prompt": "<creative brief assembled by fantasy-book-engine orchestrator>",
  "stream": false
}
```

### Responsibility Split

| Responsibility | Owner |
|---------------|-------|
| All creative content generation (prose, dialogue, plot beats, pacing, scene structure, conflict design) | narrative-forge |
| World bible storage and retrieval (PostgreSQL + pgvector) | fantasy-book-engine |
| Context assembly for creative briefs | fantasy-book-engine |
| Continuity validation against established canon | fantasy-book-engine |
| Divine constraint enforcement | fantasy-book-engine |
| Character state tracking | fantasy-book-engine |
| Series timeline and thread management | fantasy-book-engine |
| Approving or rejecting generated content | fantasy-book-engine |

## End-to-End Workflow

```bash
make setup          # Create Python venv, install dependencies, install Ollama
# Place books in data/raw/
make prepare        # Extract -> chunk -> classify -> generate training pairs
make train          # QLoRA fine-tune (~1.5-3 hours)
make evaluate       # Generate sample outputs for review
make merge          # Merge LoRA adapter into base model
make export         # Convert to GGUF
make register       # Register with Ollama as 'narrative-forge'
ollama run narrative-forge   # Test directly
```

**Iteration cycle**: To improve the model, add more books to `data/raw/`, adjust templates in `data/templates/`, tweak configs, and re-run from `make prepare`. Previous checkpoints preserved for comparison.

## Future Scaling Path

- **13B model**: Same pipeline, change `base_model` in `training_config.yaml`. Uses QLoRA across both GPUs.
- **Continual pre-training**: Add a pre-training phase before QLoRA for deeper style absorption at the vocabulary/rhythm level.
- **Multiple specialized models**: Train separate models for different narrative modes (action-heavy, introspective, dialogue-heavy) and have the orchestrator route to the appropriate one.
