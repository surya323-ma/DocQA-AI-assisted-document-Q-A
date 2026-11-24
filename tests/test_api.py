from app import config

from .conftest import make_pdf

BIO = (b"Photosynthesis lets plants turn sunlight, water and carbon dioxide into glucose.\n\n"
       b"Mitochondria produce ATP, the energy currency of the cell.")


def upload(client, name="bio.txt", data=BIO):
    return client.post("/api/documents", files={"file": (name, data)})


def test_health_and_index(client):
    assert client.get("/health").json() == {"status": "ok", "llm": "extractive"}
    assert client.get("/").status_code == 200


def test_upload_list_ask_delete(client):
    res = upload(client)
    assert res.status_code == 201 and res.json()["chunks"] >= 1
    doc_id = res.json()["id"]

    assert client.get("/api/documents").json()[0]["name"] == "bio.txt"

    answer = client.post("/api/ask", json={"question": "What do plants use sunlight for?"}).json()
    assert answer["mode"] == "extractive"
    assert answer["sources"][0]["doc"] == "bio.txt" and "sunlight" in answer["sources"][0]["text"]

    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert client.get("/api/documents").json() == []
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404


def test_pdf_upload(client):
    res = upload(client, "notes.pdf", make_pdf("Octopuses have three hearts and blue blood"))
    assert res.status_code == 201
    answer = client.post("/api/ask", json={"question": "How many hearts do octopuses have?"}).json()
    assert "three hearts" in answer["sources"][0]["text"]


def test_bad_uploads(client):
    assert upload(client, "image.png", b"123").status_code == 422
    assert upload(client, "empty.txt", b"   ").status_code == 422
    assert upload(client, "broken.pdf", b"not a pdf").status_code == 422


def test_upload_size_limit(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0)
    assert upload(client).status_code == 413


def test_ask_validation_and_empty_library(client):
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/api/ask", json={"question": "What is in my documents?"}).status_code == 409


def test_ask_can_be_limited_to_documents(client):
    a = upload(client, "bio.txt").json()["id"]
    b = upload(client, "law.txt", b"A contract needs an offer, acceptance and consideration to be binding.").json()["id"]
    res = client.post("/api/ask", json={"question": "What makes a contract binding?", "doc_ids": [b]}).json()
    assert {s["doc"] for s in res["sources"]} == {"law.txt"}
    res = client.post("/api/ask", json={"question": "What makes a contract binding?", "doc_ids": [a]}).json()
    assert res["mode"] == "none"  # nothing relevant inside the selected document
