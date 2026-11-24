"""Split text into overlapping, paragraph-aware chunks."""
import re


def normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _split_long(paragraph: str, size: int) -> list[str]:
    """Break an oversized paragraph on sentence ends, then on spaces as a last resort."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces, current = [], ""
    for sentence in sentences:
        while len(sentence) > size:
            cut = sentence.rfind(" ", 0, size)
            if cut <= 0:  # no space to break on: hard cut
                cut = size
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + len(sentence) + 1 > size:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text = normalise(text)
    if not text:
        return []
    units: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if para:
            units.extend(_split_long(para, size) if len(para) > size else [para])

    chunks, current = [], ""
    for unit in units:
        if current and len(current) + len(unit) + 2 > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail  # start on a word boundary
            current = f"{tail}\n\n{unit}".strip() if tail else unit
        else:
            current = f"{current}\n\n{unit}" if current else unit
    if current:
        chunks.append(current)
    return chunks
