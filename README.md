# DocQA: AI-assisted document Q&A

Upload PDF or text documents and ask questions about them. Answers cite the passages they came from, with your keywords highlighted so you can verify them.

**Stack:** Python 3.12 · FastAPI · scikit-learn (TF-IDF retrieval) · pypdf · LLM API (Anthropic Claude) · SQLAlchemy · pytest

## How it works
1. **Extract:** `pypdf` reads PDFs; TXT/MD are decoded as UTF-8. Scanned PDFs, encrypted PDFs and empty files return clear errors.
2. **Chunk:** `app/chunking.py` splits text into ~900-character, paragraph-aware chunks with overlap so answers aren't cut at chunk edges.
3. **Retrieve:** `app/retrieval.py` ranks chunks against the question with TF-IDF (unigrams and bigrams) and cosine similarity, and keeps the top 4 above a minimum score.
4. **Prompt:** `app/llm.py` sends only those excerpts to the model with instructions to cite `[n]`, say "not found" when the excerpts lack the answer, and ignore instructions hidden inside documents.
5. **Degrade gracefully:** with no API key, or if the API call fails, the app returns the most relevant sentences with citations instead of an error.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...                  # optional
uvicorn app.main:app --reload
```
Open http://localhost:8000. REST docs are at `/docs`.

## Tests
```bash
pytest     # 23 tests, ~94% coverage; the LLM is faked, so no key or network is needed
```

## Deploy to Render
1. Push to GitHub.
2. Render → **New → Blueprint** → select the repo.
3. Optionally paste your `ANTHROPIC_API_KEY` when asked.
4. Open the URL. Health check: `/health` (shows whether the AI key is active).

> Free Render services sleep when idle. The first request after a break is slow, and the first build downloads scikit-learn, so allow a few minutes.
> Uploaded documents live in the database. Free Postgres databases expire after a limited period; check Render's current terms. Without `DATABASE_URL` the app uses SQLite, which resets on every redeploy.

## API
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/documents` | upload a file (multipart) |
| GET | `/api/documents` | list documents |
| DELETE | `/api/documents/{id}` | remove a document |
| POST | `/api/ask` | `{"question": "...", "doc_ids": [1,2]}` returns answer and sources |

## Ideas to extend
Embedding-based retrieval (sentence-transformers or an embeddings API), per-user accounts, streaming answers, OCR for scanned PDFs.
