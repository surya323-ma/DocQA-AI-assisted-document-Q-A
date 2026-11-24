"""TF-IDF retrieval: rank chunks by cosine similarity to the question."""
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


@dataclass
class Hit:
    chunk_id: int
    doc_id: int
    doc_name: str
    text: str
    score: float


def top_chunks(question: str, chunks: list[dict], k: int = 4, min_score: float = 0.05) -> list[Hit]:
    """chunks: dicts with id, doc_id, doc_name, text. Returns up to k hits above min_score."""
    if not chunks or not question.strip():
        return []
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform(texts + [question])
    except ValueError:  # nothing but stop words
        return []
    scores = linear_kernel(matrix[-1], matrix[:-1]).ravel()
    order = scores.argsort()[::-1][:k]
    return [
        Hit(chunks[i]["id"], chunks[i]["doc_id"], chunks[i]["doc_name"], chunks[i]["text"], float(scores[i]))
        for i in order
        if scores[i] >= min_score
    ]
