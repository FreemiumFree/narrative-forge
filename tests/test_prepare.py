import json
import os

from src.prepare import run_prepare


class TestRunPrepare:
    def test_produces_jsonl_output(self, sample_raw_dir, tmp_dir):
        output_dir = os.path.join(tmp_dir, "processed")
        os.makedirs(output_dir)

        templates = {
            "prose": ["Write: {scene_hint}."],
            "dialogue": ["Write dialogue: {scene_hint}."],
            "action": ["Write action: {scene_hint}."],
            "internal": ["Write thoughts: {scene_hint}."],
            "worldbuilding": ["Write lore: {scene_hint}."],
            "transition": ["Write transition: {scene_hint}."],
        }

        stats = run_prepare(
            raw_dir=sample_raw_dir,
            output_dir=output_dir,
            templates=templates,
            min_chunk_words=10,
            max_chunk_words=200,
        )

        # Check JSONL file was created
        train_path = os.path.join(output_dir, "train.jsonl")
        assert os.path.exists(train_path)

        # Check each line is valid JSON with required fields
        with open(train_path) as f:
            for line in f:
                record = json.loads(line)
                assert "instruction" in record
                assert "completion" in record
                assert "type" in record

        # Check stats
        assert stats["total_pairs"] > 0
        assert stats["total_books"] == 1
        assert "type_breakdown" in stats

    def test_creates_train_and_val_splits(self, sample_raw_dir, tmp_dir):
        output_dir = os.path.join(tmp_dir, "processed")
        os.makedirs(output_dir)

        templates = {
            "prose": ["Write: {scene_hint}."],
            "dialogue": ["Write dialogue: {scene_hint}."],
            "action": ["Write action: {scene_hint}."],
            "internal": ["Write thoughts: {scene_hint}."],
            "worldbuilding": ["Write lore: {scene_hint}."],
            "transition": ["Write transition: {scene_hint}."],
        }

        run_prepare(
            raw_dir=sample_raw_dir,
            output_dir=output_dir,
            templates=templates,
            min_chunk_words=10,
            max_chunk_words=200,
        )

        train_path = os.path.join(output_dir, "train.jsonl")
        val_path = os.path.join(output_dir, "val.jsonl")
        assert os.path.exists(train_path)
        assert os.path.exists(val_path)

    def test_writes_stats_file(self, sample_raw_dir, tmp_dir):
        output_dir = os.path.join(tmp_dir, "processed")
        os.makedirs(output_dir)

        templates = {
            "prose": ["Write: {scene_hint}."],
            "dialogue": ["Write dialogue: {scene_hint}."],
            "action": ["Write action: {scene_hint}."],
            "internal": ["Write thoughts: {scene_hint}."],
            "worldbuilding": ["Write lore: {scene_hint}."],
            "transition": ["Write transition: {scene_hint}."],
        }

        run_prepare(
            raw_dir=sample_raw_dir,
            output_dir=output_dir,
            templates=templates,
            min_chunk_words=10,
            max_chunk_words=200,
        )

        stats_path = os.path.join(output_dir, "stats.json")
        assert os.path.exists(stats_path)

        with open(stats_path) as f:
            stats = json.load(f)
        assert "total_pairs" in stats
        assert "total_books" in stats
