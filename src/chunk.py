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
        para_word_list = para.split()
        para_words = len(para_word_list)

        # If a single paragraph exceeds max_words, split it into sub-chunks
        if para_words > max_words:
            # Flush any pending chunk first
            if current_chunk and current_words >= min_words:
                chunks.append({
                    "text": "\n\n".join(current_chunk),
                    "word_count": current_words,
                })
            current_chunk = []
            current_words = 0
            # Split paragraph into word-count slices
            for i in range(0, para_words, max_words):
                slice_words = para_word_list[i:i + max_words]
                slice_text = " ".join(slice_words)
                slice_count = len(slice_words)
                if slice_count >= min_words:
                    chunks.append({
                        "text": slice_text,
                        "word_count": slice_count,
                    })
            continue

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
