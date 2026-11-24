from app.retrieval import top_chunks

CHUNKS = [
    {"id": 1, "doc_id": 1, "doc_name": "bio.txt", "text": "Photosynthesis lets plants turn sunlight, water and carbon dioxide into glucose."},
    {"id": 2, "doc_id": 1, "doc_name": "bio.txt", "text": "Mitochondria produce ATP, the energy currency of the cell."},
    {"id": 3, "doc_id": 2, "doc_name": "law.txt", "text": "A contract needs an offer, acceptance and consideration to be binding."},
]


def test_most_relevant_chunk_first():
    hits = top_chunks("How do plants use sunlight?", CHUNKS)
    assert hits and hits[0].chunk_id == 1 and hits[0].doc_name == "bio.txt"


def test_other_topic_ranks_other_chunk():
    assert top_chunks("What makes a contract binding?", CHUNKS)[0].chunk_id == 3


def test_irrelevant_question_returns_nothing():
    assert top_chunks("quantum chromodynamics lagrangian", CHUNKS) == []


def test_edge_cases():
    assert top_chunks("anything", []) == []
    assert top_chunks("   ", CHUNKS) == []
    assert top_chunks("the and of", CHUNKS) == []  # only stop words


def test_k_limits_results():
    assert len(top_chunks("energy cell plants contract", CHUNKS, k=2, min_score=0)) == 2
