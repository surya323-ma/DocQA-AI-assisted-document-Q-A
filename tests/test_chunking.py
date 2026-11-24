import pytest

from app.chunking import chunk_text


def test_empty_text():
    assert chunk_text("   \n\n ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.") == ["Hello world."]


def test_chunks_respect_size_and_overlap():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(12))
    chunks = chunk_text(text, size=400, overlap=80)
    assert len(chunks) > 3
    assert all(len(c) <= 400 + 80 + 2 for c in chunks)
    # overlap: the start of each chunk repeats words from the end of the previous one
    first_words = chunks[1].split()[:3]
    assert all(w in chunks[0] for w in first_words)


def test_long_paragraph_is_split_on_sentences():
    text = " ".join(f"Sentence number {i} is here." for i in range(80))
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 5 and all(len(c) < 420 for c in chunks)


def test_unbroken_text_is_still_split():
    chunks = chunk_text("x" * 2500, size=500, overlap=50)
    assert len(chunks) >= 5


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("abc", size=100, overlap=100)
