import os

from src.extract import extract_text, extract_from_directory


class TestExtractText:
    def test_extracts_from_txt(self, sample_txt_file):
        result = extract_text(sample_txt_file)
        assert "forest stretched endlessly" in result
        assert "Sera pressed her palm" in result

    def test_strips_chapter_headers(self, sample_txt_file):
        result = extract_text(sample_txt_file)
        assert "Chapter 1" in result

    def test_returns_empty_for_missing_file(self, tmp_dir):
        result = extract_text(os.path.join(tmp_dir, "nonexistent.txt"))
        assert result == ""

    def test_handles_utf8_encoding(self, tmp_dir):
        path = os.path.join(tmp_dir, "unicode.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("The na\u00efve hero walked through the caf\u00e9.")
        result = extract_text(path)
        assert "na\u00efve" in result
        assert "caf\u00e9" in result


class TestExtractFromDirectory:
    def test_finds_all_txt_files(self, sample_raw_dir):
        results = extract_from_directory(sample_raw_dir)
        assert len(results) == 1
        assert results[0]["source"] == "sample_book.txt"
        assert "forest stretched endlessly" in results[0]["text"]

    def test_ignores_unsupported_formats(self, sample_raw_dir):
        with open(os.path.join(sample_raw_dir, "notes.json"), "w") as f:
            f.write("{}")
        results = extract_from_directory(sample_raw_dir)
        assert len(results) == 1

    def test_empty_directory(self, tmp_dir):
        empty = os.path.join(tmp_dir, "empty")
        os.makedirs(empty)
        results = extract_from_directory(empty)
        assert results == []
