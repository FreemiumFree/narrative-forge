import os
import tempfile

import pytest


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_txt_file(tmp_dir):
    path = os.path.join(tmp_dir, "sample_book.txt")
    content = (
        "Chapter 1: The Beginning\n\n"
        "The forest stretched endlessly before her, ancient trees reaching "
        "toward a sky heavy with storm clouds. She could smell the rain coming, "
        "a clean sharp scent that cut through the decay of fallen leaves.\n\n"
        '"We should turn back," Marcus said, his hand on his sword hilt.\n\n'
        '"No." Sera kept walking. "The temple is close. I can feel it."\n\n'
        "The ground trembled beneath their feet. Somewhere deep below, "
        "something was waking.\n\n"
        "Chapter 2: The Temple\n\n"
        "They found it at dawn — a structure of black obsidian rising from "
        "the forest floor like a broken tooth. Equations were carved into "
        "every surface, mathematical truths that predated language itself.\n\n"
        "Sera pressed her palm against the stone. It was warm.\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


@pytest.fixture
def sample_raw_dir(tmp_dir, sample_txt_file):
    """A raw directory containing one sample book."""
    raw_dir = os.path.join(tmp_dir, "raw")
    os.makedirs(raw_dir)
    import shutil
    shutil.copy(sample_txt_file, os.path.join(raw_dir, "sample_book.txt"))
    return raw_dir
