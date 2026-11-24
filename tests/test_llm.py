from types import SimpleNamespace

from app import config, llm
from app.retrieval import Hit

HITS = [Hit(1, 1, "bio.txt", "Photosynthesis lets plants turn sunlight and water into glucose and oxygen.", 0.6)]


class FakeClient:
    def __init__(self, text="Plants make glucose [1].", boom=False):
        self.calls, self._text, self._boom = [], text, boom
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._boom:
            raise RuntimeError("api down")
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._text)])


def test_no_hits():
    text, mode = llm.answer("anything?", [])
    assert mode == "none" and "couldn't find" in text


def test_extractive_fallback_without_key():
    text, mode = llm.answer("What do plants turn into glucose?", HITS)
    assert mode == "extractive" and "[1]" in text and "sunlight" in text


def test_llm_path_builds_grounded_prompt():
    client = FakeClient()
    text, mode = llm.answer("What do plants make?", HITS, client=client)
    assert mode == "llm" and text == "Plants make glucose [1]."
    call = client.calls[0]
    assert call["model"] == config.LLM_MODEL
    assert "only the numbered excerpts" in call["system"]
    assert "<excerpts>" in call["messages"][0]["content"] and "bio.txt" in call["messages"][0]["content"]


def test_llm_error_degrades_to_extractive():
    text, mode = llm.answer("What do plants turn into glucose?", HITS, client=FakeClient(boom=True))
    assert mode == "extractive"


def test_extractive_without_overlap():
    hits = [Hit(1, 1, "x.txt", "Completely unrelated text about something else entirely here.", 0.1)]
    assert "couldn't find" in llm.extractive_answer("zebra migration", hits)
