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
