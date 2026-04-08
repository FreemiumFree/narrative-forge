"""Analyze book passages with Claude Opus to extract writing craft techniques.

For each passage, identifies the craft techniques being used — not WHAT happens,
but HOW the author achieves the effect. This produces a catalog of reusable
writing techniques that can be taught to another model.
"""
import json
import os
import time

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

MODEL = "claude-opus-4-20250514"

ANALYSIS_PROMPT = """You are a master-level literary craft analyst. Your job is to analyze a passage from a fantasy novel and extract the WRITING TECHNIQUES being used — not the plot, characters, or world, but the CRAFT.

Analyze this passage and identify:

1. **Primary Technique**: The dominant craft technique (e.g., "tension through environmental foreshadowing", "character revelation through constrained dialogue", "world-building through sensory immersion")

2. **How It Works**: A 2-3 sentence explanation of the mechanical technique — what the author does with sentence structure, information flow, pacing, word choice, or narrative perspective to achieve the effect. Be specific about the craft, not the content.

3. **Secondary Techniques** (if any): Other craft techniques present in the passage (list 0-3).

4. **Style Elements**: Specific prose style observations — sentence length patterns, vocabulary register, sensory channels used, metaphor density, dialogue-to-narration ratio.

5. **Emotional Effect**: What emotional or narrative effect these techniques achieve for the reader.

6. **Technique Category**: Classify as one of: prose_style, dialogue_craft, action_choreography, worldbuilding_technique, character_interiority, pacing_structure, tension_building, atmospheric_description, transition_craft, lore_integration

IMPORTANT: Do NOT reference specific character names, place names, or plot events from the source material. Describe the techniques in universal, reusable terms.

Respond in JSON format:
{
  "primary_technique": "...",
  "how_it_works": "...",
  "secondary_techniques": ["...", "..."],
  "style_elements": "...",
  "emotional_effect": "...",
  "category": "..."
}

PASSAGE:
"""

def analyze_chunk(client: Anthropic, chunk_text: str, source: str = "") -> dict | None:
    """Analyze a single text chunk to extract craft techniques.

    Returns a dict with technique analysis, or None if the chunk isn't useful.
    """
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": ANALYSIS_PROMPT + chunk_text
            }]
        )

        text = resp.content[0].text.strip()

        # Parse JSON from response (handle markdown code blocks)
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        analysis = json.loads(text)
        analysis["source"] = source
        return analysis

    except (json.JSONDecodeError, IndexError, KeyError) as e:
        print(f"    Warning: Failed to parse analysis: {e}")
        return None
    except Exception as e:
        print(f"    Warning: API error: {e}")
        time.sleep(2)
        return None


def analyze_chunks(
    chunks: list[dict],
    output_path: str,
    max_chunks_per_book: int = 150,
) -> list[dict]:
    """Analyze multiple chunks and save results incrementally.

    Args:
        chunks: List of {"text": str, "type": str, "source": str}
        output_path: Path to save analysis results (JSONL, appended incrementally)
        max_chunks_per_book: Max chunks to analyze per source book

    Returns:
        List of analysis dicts
    """
    client = Anthropic()
    analyses = []

    # Load existing analyses if resuming
    existing = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                record = json.loads(line)
                existing.add(record.get("_chunk_hash", ""))
                analyses.append(record)
        print(f"  Resuming: {len(analyses)} existing analyses found")

    # Group chunks by source and limit per book
    from collections import defaultdict
    by_source = defaultdict(list)
    for chunk in chunks:
        by_source[chunk["source"]].append(chunk)

    total_to_analyze = 0
    for source, source_chunks in by_source.items():
        total_to_analyze += min(len(source_chunks), max_chunks_per_book)

    analyzed = len(analyses)
    print(f"  Total chunks to analyze: {total_to_analyze}")

    with open(output_path, "a", encoding="utf-8") as f:
        for source, source_chunks in sorted(by_source.items()):
            # Limit chunks per book
            selected = source_chunks[:max_chunks_per_book]
            print(f"  Analyzing {source} ({len(selected)} chunks)...")

            for i, chunk in enumerate(selected):
                # Create a hash for deduplication
                chunk_hash = str(hash(chunk["text"][:200]))
                if chunk_hash in existing:
                    continue

                analysis = analyze_chunk(client, chunk["text"], source)

                if analysis:
                    analysis["_chunk_hash"] = chunk_hash
                    analysis["chunk_type"] = chunk.get("type", "unknown")
                    analyses.append(analysis)
                    f.write(json.dumps(analysis, ensure_ascii=False) + "\n")
                    f.flush()

                analyzed += 1
                if analyzed % 25 == 0:
                    print(f"    Progress: {analyzed}/{total_to_analyze} chunks analyzed")

                # Small delay to avoid rate limits
                time.sleep(0.5)

    print(f"  Analysis complete: {len(analyses)} techniques cataloged")
    return analyses
