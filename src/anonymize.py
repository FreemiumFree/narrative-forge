"""Anonymize book text by replacing character names, places, and world-specific terms.

Uses a hybrid approach:
1. spaCy NER for known entity types
2. Pattern matching for capitalized proper nouns that NER misses (common with fantasy names)

Preserves prose style, narrative structure, and world-building patterns
while stripping specific character identities and proper nouns.
"""
import re
from collections import Counter

import spacy

_nlp = None

# Common words that are capitalized but aren't proper nouns — don't replace these
COMMON_TITLE_CASE = {
    "The", "A", "An", "And", "But", "Or", "For", "Nor", "So", "Yet",
    "In", "On", "At", "To", "By", "Of", "Up", "As", "If", "It",
    "Is", "Was", "Are", "Were", "Be", "Do", "Did", "Has", "Had",
    "He", "She", "His", "Her", "They", "Them", "Their", "Its",
    "We", "You", "I", "My", "Our", "Your", "Who", "What", "When",
    "Where", "Why", "How", "That", "This", "These", "Those",
    "Not", "No", "Yes", "All", "Each", "Every", "Some", "Any",
    "With", "From", "Into", "Through", "During", "Before", "After",
    "Above", "Below", "Between", "Under", "Over", "Out", "Off",
    "Then", "Than", "Also", "Just", "Only", "Still", "Even", "Now",
    "Here", "There", "Very", "Too", "Much", "Many", "Few", "More",
    "Most", "Other", "Such", "Like", "Well", "Back", "Long", "Good",
    "New", "Old", "Great", "High", "Little", "Own", "Right", "Big",
    "Small", "Young", "Last", "Next", "First", "Second", "Third",
    "Part", "Chapter", "Book", "One", "Two", "Three", "Four", "Five",
    "Six", "Seven", "Eight", "Nine", "Ten", "Hundred", "Thousand",
    "King", "Queen", "Prince", "Princess", "Lord", "Lady", "Sir",
    "God", "Gods", "Master", "Father", "Mother", "Brother", "Sister",
    "Captain", "Commander", "General", "Knight", "Soldier", "Guard",
    "North", "South", "East", "West", "Storm", "Dark", "Light",
    "Fire", "Water", "Earth", "Wind", "Stone", "Iron", "Gold",
    "Silver", "Black", "White", "Red", "Blue", "Green",
    "Perhaps", "Indeed", "However", "Though", "Although", "Because",
    "Something", "Nothing", "Everything", "Someone", "Anyone",
    "Enough", "Already", "Always", "Never", "Again", "Once",
    "Could", "Would", "Should", "Might", "Must", "Shall", "Will",
    "Don", "Didn", "Doesn", "Won", "Wouldn", "Couldn", "Shouldn",
}

# Generic replacements — rotates through to maintain distinctness
PERSON_TAGS = [
    "the warrior", "the woman", "the man", "the elder",
    "the youth", "the stranger", "the captain", "the scholar",
    "the healer", "the knight", "the mage", "the scout",
    "the commander", "the traveler", "the prince", "the queen",
    "the soldier", "the priest", "the merchant", "the sage",
]

PLACE_TAGS = [
    "the city", "the kingdom", "the fortress", "the village",
    "the tower", "the temple", "the ruins", "the valley",
    "the highlands", "the capital", "the outpost", "the sanctuary",
    "the plains", "the peaks", "the coast", "the stronghold",
]

THING_TAGS = [
    "the power", "the force", "the energy", "the essence",
    "the artifact", "the relic", "the bond", "the gift",
]


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _find_proper_nouns(text: str) -> set[str]:
    """Find capitalized words/phrases that are likely proper nouns.

    Catches fantasy names that NER misses by looking for:
    - Capitalized words not at the start of sentences
    - Multi-word capitalized phrases (e.g. "Shattered Plains")
    """
    proper_nouns = set()

    # Find capitalized words that aren't sentence starters
    # Split into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sentence in sentences:
        words = sentence.split()
        for i, word in enumerate(words):
            # Skip first word of sentence
            if i == 0:
                continue
            # Skip words after quotation marks (dialogue starts)
            if i > 0 and words[i-1] in ('"', "'", "\u201c", "\u2018"):
                continue

            # Clean punctuation from word
            clean = re.sub(r'[^a-zA-Z\'-]', '', word)

            if not clean:
                continue

            # Check if it's a capitalized word that's not common
            if clean[0].isupper() and clean not in COMMON_TITLE_CASE:
                # Check if it's followed by another capitalized word (multi-word name)
                if i + 1 < len(words):
                    next_clean = re.sub(r'[^a-zA-Z\'-]', '', words[i+1])
                    if next_clean and next_clean[0].isupper() and next_clean not in COMMON_TITLE_CASE:
                        proper_nouns.add(f"{clean} {next_clean}")

                proper_nouns.add(clean)

    return proper_nouns


def _classify_proper_noun(word: str, context: str) -> str:
    """Guess whether a proper noun is a person, place, or thing based on context."""
    word_lower = word.lower()
    ctx = context.lower()

    # Look for context clues around the word
    # Person clues: preceded/followed by said, asked, looked, etc.
    person_patterns = [
        rf'{re.escape(word)}\s+(said|asked|replied|whispered|shouted|nodded|shook|looked|turned|stood|walked|ran|sat|felt|thought|knew|wanted|smiled|frowned|sighed)',
        rf'(said|asked|replied|told)\s+{re.escape(word)}',
        rf'{re.escape(word)}\'s\s+(eyes|face|hand|voice|arm|heart|mind|head|sword|blade)',
    ]
    for pattern in person_patterns:
        if re.search(pattern, context, re.IGNORECASE):
            return "person"

    # Place clues
    place_patterns = [
        rf'(in|at|to|from|toward|towards|across|through|of)\s+{re.escape(word)}',
        rf'{re.escape(word)}\s+(city|kingdom|plains|mountains|forest|tower|temple|fortress)',
    ]
    for pattern in place_patterns:
        if re.search(pattern, context, re.IGNORECASE):
            return "place"

    # Default to person (most common in fiction)
    return "person"


def anonymize_text(text: str) -> str:
    """Replace named entities and proper nouns with generic placeholders."""
    nlp = _get_nlp()

    if len(text) > 900000:
        nlp.max_length = len(text) + 1000

    # Step 1: Get NER entities
    doc = nlp(text)
    entity_map = {}
    person_idx = 0
    place_idx = 0
    thing_idx = 0

    for ent in doc.ents:
        if ent.text in entity_map:
            continue
        if ent.label_ in ("PERSON",):
            entity_map[ent.text] = PERSON_TAGS[person_idx % len(PERSON_TAGS)]
            person_idx += 1
        elif ent.label_ in ("GPE", "LOC", "FAC"):
            entity_map[ent.text] = PLACE_TAGS[place_idx % len(PLACE_TAGS)]
            place_idx += 1
        elif ent.label_ == "ORG":
            entity_map[ent.text] = PLACE_TAGS[place_idx % len(PLACE_TAGS)]
            place_idx += 1

    # Step 2: Find proper nouns that NER missed
    proper_nouns = _find_proper_nouns(text)

    for noun in proper_nouns:
        if noun in entity_map:
            continue
        # Skip if it's a substring of an already-mapped entity
        if any(noun in ent for ent in entity_map):
            continue

        category = _classify_proper_noun(noun, text)
        if category == "person":
            entity_map[noun] = PERSON_TAGS[person_idx % len(PERSON_TAGS)]
            person_idx += 1
        elif category == "place":
            entity_map[noun] = PLACE_TAGS[place_idx % len(PLACE_TAGS)]
            place_idx += 1
        else:
            entity_map[noun] = THING_TAGS[thing_idx % len(THING_TAGS)]
            thing_idx += 1

    # Step 3: Apply replacements — longest first to avoid partial matches
    result = text
    for entity, replacement in sorted(entity_map.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = re.compile(re.escape(entity))
        result = pattern.sub(replacement, result)

    return result


def anonymize_chunk(chunk_text: str) -> str:
    """Anonymize a single chunk of text."""
    return anonymize_text(chunk_text)
