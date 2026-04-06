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
