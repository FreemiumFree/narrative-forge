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
