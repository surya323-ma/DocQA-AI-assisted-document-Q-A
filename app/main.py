from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import config, llm
from .chunking import chunk_text
from .database import Base, Chunk, Document, engine, get_db
from .extract import ExtractError, extract_text
from .retrieval import top_chunks

STATIC = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(engine)
    yield


app = FastAPI(title="DocQA", description="Ask questions about your documents.", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    doc_ids: list[int] | None = None


@app.get("/health")
def health():
    return {"status": "ok", "llm": "claude" if config.ANTHROPIC_API_KEY else "extractive"}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/documents")
def list_documents(db: Session = Depends(get_db)):
    rows = db.execute(
        select(Document.id, Document.name, Document.chars, func.count(Chunk.id))
        .join(Chunk, Chunk.doc_id == Document.id, isouter=True)
        .group_by(Document.id)
        .order_by(Document.id.desc())
    ).all()
    return [{"id": i, "name": n, "chars": c, "chunks": k} for i, n, c, k in rows]


@app.post("/api/documents", status_code=201)
async def upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    data = await file.read(config.MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {config.MAX_UPLOAD_MB} MB.")
    try:
        text = extract_text(file.filename or "", data)
    except ExtractError as exc:
        raise HTTPException(422, str(exc))
    pieces = chunk_text(text, config.CHUNK_CHARS, config.CHUNK_OVERLAP)
    doc = Document(name=(file.filename or "document")[:200], chars=len(text))
    doc.chunks = [Chunk(position=i, text=p) for i, p in enumerate(pieces)]
    db.add(doc)
    db.commit()
    return {"id": doc.id, "name": doc.name, "chars": doc.chars, "chunks": len(pieces)}


@app.delete("/api/documents/{doc_id}", status_code=204)
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    db.delete(doc)
    db.commit()


@app.post("/api/ask")
def ask(body: AskRequest, db: Session = Depends(get_db)):
    stmt = select(Chunk.id, Chunk.doc_id, Document.name, Chunk.text).join(Document, Document.id == Chunk.doc_id)
    if body.doc_ids:
        stmt = stmt.where(Chunk.doc_id.in_(body.doc_ids))
    chunks = [{"id": i, "doc_id": d, "doc_name": n, "text": t} for i, d, n, t in db.execute(stmt).all()]
    if not chunks:
        raise HTTPException(409, "Upload a document first.")
    hits = top_chunks(body.question, chunks, config.TOP_K, config.MIN_SCORE)
    text, mode = llm.answer(body.question, hits)
    return {
        "answer": text,
        "mode": mode,
        "sources": [{"n": n, "doc": h.doc_name, "score": round(h.score, 3), "text": h.text} for n, h in enumerate(hits, 1)],
    }
