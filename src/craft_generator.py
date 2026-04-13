"""Generate original training examples that demonstrate identified craft techniques.

Takes the craft analysis catalog and produces original fantasy prose that
demonstrates each technique in a fresh context — no borrowed characters,
places, or plot elements from the source books.
"""
import json
import os
import time
import random

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

MODEL = "claude-sonnet-4-5"

GENERATION_PROMPT = """You are a master fantasy fiction writer. Your task is to write an ORIGINAL passage that demonstrates a specific writing craft technique. The passage must be entirely original — new characters, new world, new situation. Do not reference any existing books, series, or authors.

CRAFT TECHNIQUE TO DEMONSTRATE:
{technique}

HOW THIS TECHNIQUE WORKS:
{how_it_works}

STYLE ELEMENTS TO INCORPORATE:
{style_elements}

TARGET EMOTIONAL EFFECT:
{emotional_effect}

PASSAGE TYPE: {category}

CONTEXT FOR THE PASSAGE:
{context}

INSTRUCTIONS:
- Write 150-350 words of original fantasy prose demonstrating this technique
- Create original characters (use generic names or titles, not names from any published work)
- Create an original setting and situation
- Focus on executing the TECHNIQUE brilliantly, not on complex plot
- The prose should feel like it belongs in a published fantasy novel
- Do NOT include any meta-commentary or explanation — just write the passage

Write the passage now:"""

# Diverse contexts to ensure variety in generated examples
CONTEXTS = {
    "prose_style": [
        "A warrior arriving at an ancient city for the first time",
        "A storm approaching across a vast plain",
        "A library containing forbidden knowledge",
        "A forest that has been corrupted by dark magic",
        "The aftermath of a great battle, seen at dawn",
        "A ship approaching an island shrouded in mist",
        "A throne room during a tense audience with the sovereign",
        "An underground cavern with bioluminescent flora",
        "A marketplace in a city where magic is traded openly",
        "A mountain pass during the first snow of winter",
        "The ruins of a civilization far older than memory",
        "A garden that exists between two realms",
    ],
    "dialogue_craft": [
        "A mentor and student disagreeing about the use of power",
        "Two rivals forced to cooperate against a common threat",
        "A leader breaking difficult news to their people",
        "A reunion between two people separated by years of conflict",
        "A negotiation between representatives of hostile nations",
        "A confession of betrayal between close allies",
        "A teacher explaining a dangerous truth to an unprepared student",
        "Two strangers discovering they share a hidden connection",
        "A commander questioning a soldier's loyalty",
        "A healer refusing to save someone who deserves death",
    ],
    "action_choreography": [
        "A duel between a magic user and a warrior who nullifies magic",
        "A chase through a city built vertically on cliff faces",
        "A siege where the defenders use unconventional tactics",
        "A fight in a collapsing temple",
        "An ambush in a canyon during a sandstorm",
        "A battle on the deck of a ship during a supernatural storm",
        "A confrontation in a place where gravity shifts unpredictably",
        "A desperate defense of a bridge against overwhelming numbers",
        "An assassination attempt during a royal ceremony",
        "A magical duel where the environment itself is a weapon",
    ],
    "worldbuilding_technique": [
        "Revealing how a magic system works through a character learning it",
        "Showing the consequences of a world's magical laws on daily life",
        "Describing a religious ceremony that reveals the nature of the gods",
        "Showing how different cultures interpret the same magical phenomenon",
        "Revealing the cost of magic through a character paying it",
        "Describing an ecosystem shaped by supernatural forces",
        "Showing how political power interacts with magical power",
        "Revealing ancient history through architectural details",
        "Describing a craft or trade unique to this world",
        "Showing how common people live with extraordinary forces around them",
    ],
    "character_interiority": [
        "A leader doubting their right to command after a costly victory",
        "A young person realizing their mentor is fallible",
        "Someone making peace with a power they never wanted",
        "A warrior struggling with the person they've become",
        "Someone recognizing their enemy's humanity for the first time",
        "A person choosing duty over personal desire",
        "Someone confronting a truth they've been avoiding",
        "A character finding unexpected strength in vulnerability",
        "Someone grieving while needing to remain strong",
        "A person realizing they've become the thing they fought against",
    ],
    "pacing_structure": [
        "The calm before a major confrontation",
        "A moment of levity between heavy scenes",
        "Building from quiet observation to sudden action",
        "A slow revelation that recontextualizes everything before it",
        "Intercutting between simultaneous events at different scales",
        "A gradual escalation from unease to full terror",
        "The decompression after a climactic event",
        "Time slowing during a pivotal moment of decision",
        "A montage of small moments building toward significance",
        "Tension held through mundane activity while danger approaches",
    ],
    "tension_building": [
        "A character walking into a situation they don't realize is dangerous",
        "A conversation where both parties know something the other doesn't",
        "The discovery that a safe place has been compromised",
        "A countdown to an event that cannot be stopped",
        "A character making a choice with unknown consequences",
        "Growing signs that something ancient has awakened",
        "A feast where hidden enemies sit among allies",
        "A journey through territory where the rules of reality are different",
        "A ritual that begins to go wrong in subtle ways",
        "The moment when a character realizes they've been betrayed",
    ],
    "atmospheric_description": [
        "A city at the boundary between two different kinds of magic",
        "A battlefield being reclaimed by nature years after the war",
        "A sacred place that inspires both reverence and fear",
        "A workshop where impossible things are crafted",
        "A border town caught between two warring cultures",
        "A place where time moves differently",
        "A prison designed to hold something that isn't human",
        "A road that travelers are warned never to walk at night",
        "A library that is alive in some fundamental way",
        "The last surviving remnant of a destroyed civilization",
    ],
    "transition_craft": [
        "Moving from the end of one character's chapter to another's beginning",
        "Shifting from an intimate scene to a wide political view",
        "Bridging a time skip while maintaining emotional continuity",
        "Transitioning from action to aftermath",
        "Moving from a flashback to the present with thematic resonance",
        "Shifting perspective from a leader to the people affected by their decision",
        "Transitioning from hope to dread through environmental cues",
        "Bridging two storylines that are about to converge",
        "Moving from a personal loss to a wider view of what was gained",
        "Transitioning between scales — from cosmic to intimate",
    ],
    "lore_integration": [
        "Ancient prophecy being reinterpreted in light of current events",
        "A myth that turns out to be a distorted historical record",
        "Religious doctrine that conceals a practical truth about magic",
        "A children's song that contains a warning about real dangers",
        "Historical architecture revealing forgotten knowledge",
        "A cultural practice that exists for reasons nobody remembers",
        "Legends about the gods that contradict each other meaningfully",
        "A scholar discovering that two separate myths describe the same event",
        "Trade goods that reveal the existence of a hidden civilization",
        "A naming convention that encodes the magical properties of things",
    ],
}


def generate_example(
    client: Anthropic,
    analysis: dict,
    context: str,
) -> dict | None:
    """Generate a single original training example from a craft analysis."""
    try:
        prompt = GENERATION_PROMPT.format(
            technique=analysis["primary_technique"],
            how_it_works=analysis["how_it_works"],
            style_elements=analysis.get("style_elements", "No specific style notes"),
            emotional_effect=analysis.get("emotional_effect", "Engage and immerse the reader"),
            category=analysis.get("category", "prose_style"),
            context=context,
        )

        resp = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )

        passage = resp.content[0].text.strip()

        # Build the instruction — craft-focused, not content-focused
        instruction = _build_instruction(analysis, context)

        return {
            "instruction": instruction,
            "completion": passage,
            "type": analysis.get("category", "prose_style"),
            "technique": analysis["primary_technique"],
        }

    except Exception as e:
        print(f"    Warning: Generation failed: {e}")
        time.sleep(2)
        return None


def _build_instruction(analysis: dict, context: str) -> str:
    """Build a craft-focused instruction for the training pair."""
    category = analysis.get("category", "prose_style")

    templates = {
        "prose_style": [
            f"Write a fantasy passage using this technique: {analysis['primary_technique']}. Context: {context}.",
            f"Demonstrate this prose craft: {analysis['primary_technique']}. Scene: {context}.",
            f"Write a passage that achieves this effect: {analysis.get('emotional_effect', 'immersion')}. Setting: {context}.",
        ],
        "dialogue_craft": [
            f"Write dialogue that demonstrates: {analysis['primary_technique']}. Situation: {context}.",
            f"Craft a dialogue scene using this technique: {analysis['primary_technique']}. Context: {context}.",
        ],
        "action_choreography": [
            f"Write an action scene using this technique: {analysis['primary_technique']}. Scenario: {context}.",
            f"Demonstrate this combat writing craft: {analysis['primary_technique']}. Setting: {context}.",
        ],
        "worldbuilding_technique": [
            f"Write a passage that builds the world using this technique: {analysis['primary_technique']}. Context: {context}.",
            f"Demonstrate this world-building craft: {analysis['primary_technique']}. Scenario: {context}.",
        ],
        "character_interiority": [
            f"Write internal monologue using this technique: {analysis['primary_technique']}. Situation: {context}.",
            f"Show a character's inner world using: {analysis['primary_technique']}. Context: {context}.",
        ],
        "pacing_structure": [
            f"Write a passage with this pacing technique: {analysis['primary_technique']}. Scene: {context}.",
            f"Demonstrate this pacing craft: {analysis['primary_technique']}. Context: {context}.",
        ],
        "tension_building": [
            f"Build tension using this technique: {analysis['primary_technique']}. Situation: {context}.",
            f"Write a tense passage using: {analysis['primary_technique']}. Context: {context}.",
        ],
        "atmospheric_description": [
            f"Create atmosphere using this technique: {analysis['primary_technique']}. Setting: {context}.",
            f"Write atmospheric prose using: {analysis['primary_technique']}. Location: {context}.",
        ],
        "transition_craft": [
            f"Write a transition using this technique: {analysis['primary_technique']}. Context: {context}.",
            f"Demonstrate this transition craft: {analysis['primary_technique']}. Scenario: {context}.",
        ],
        "lore_integration": [
            f"Weave lore into narrative using: {analysis['primary_technique']}. Context: {context}.",
            f"Write a passage that integrates lore using: {analysis['primary_technique']}. Setting: {context}.",
        ],
    }

    options = templates.get(category, templates["prose_style"])
    return random.choice(options)


def generate_training_data(
    analyses: list[dict],
    output_path: str,
    examples_per_technique: int = 3,
    seed: int = 42,
) -> list[dict]:
    """Generate original training examples for all analyzed techniques.

    Args:
        analyses: List of craft analysis dicts from craft_analyzer
        output_path: Path to save generated training pairs (JSONL)
        examples_per_technique: Number of original examples per technique
        seed: Random seed

    Returns:
        List of training pair dicts
    """
    random.seed(seed)
    client = Anthropic()
    examples = []

    # Load existing examples if resuming
    existing_count = 0
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                examples.append(json.loads(line))
                existing_count += 1
        print(f"  Resuming: {existing_count} existing examples found")

    total_to_generate = len(analyses) * examples_per_technique - existing_count
    print(f"  Generating {total_to_generate} original examples from {len(analyses)} techniques...")

    generated = existing_count
    with open(output_path, "a", encoding="utf-8") as f:
        for i, analysis in enumerate(analyses):
            if i * examples_per_technique < existing_count:
                continue

            category = analysis.get("category", "prose_style")
            contexts = CONTEXTS.get(category, CONTEXTS["prose_style"])

            # Pick diverse contexts for this technique
            selected_contexts = random.sample(
                contexts, min(examples_per_technique, len(contexts))
            )

            for context in selected_contexts:
                example = generate_example(client, analysis, context)

                if example:
                    examples.append(example)
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    f.flush()

                generated += 1
                if generated % 25 == 0:
                    print(f"    Progress: {generated}/{total_to_generate + existing_count} examples generated")

                time.sleep(0.5)

    print(f"  Generation complete: {len(examples)} training examples")
    return examples
