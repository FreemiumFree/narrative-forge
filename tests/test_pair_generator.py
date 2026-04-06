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
