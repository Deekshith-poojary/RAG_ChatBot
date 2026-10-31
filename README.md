# PDF RAG Chatbot

Chat with large private PDF collections. Open-source stack, fully local:

| Layer | Choice |
|---|---|
| PDF text | PyMuPDF (native text layer) |
| OCR | PaddleOCR - scanned pages (full-page) and large embedded images |
| Chunking | LangChain `RecursiveCharacterTextSplitter`, token-counted with the embedding tokenizer (500 tokens, 100 overlap) |
| Embeddings | `BAAI/bge-small-en-v1.5` (sentence-transformers) |
| Vector DB | Qdrant, embedded local file mode (`data/qdrant`), HNSW index, cosine |
| Reranker | `BAAI/bge-reranker-base` (FlagEmbedding), optional |
| LLM | Your Ollama endpoint, wrapped by FastAPI |
| UI | Plain HTML/CSS/JS (ES modules), served by the same FastAPI app |

## Layout

```
src/rag_chatbot/
  config.py            settings (env / .env)
  domain.py            shared types (PageContent, Chunk, Source, DocumentRecord, ...)
  ingestion/           extractor (PyMuPDF+OCR) -> cleaning -> chunker -> pipeline; registry; service
  embeddings/          sentence-transformers wrapper
  retrieval/           Qdrant store, reranker, retriever
  generation/          Ollama client, prompts, citation check, RAG service
  api/                 FastAPI app, container (wiring), routes/, web.py (serves the UI)
  web/static/          index.html, css/styles.css, js/ (api, chat, documents, settings, ...)
```

## Setup

```powershell
.venv\Scripts\activate
pip install -e . --no-deps        # dependencies are already installed
pip install langdetect            # language detection (new dependency)
copy .env.example .env            # set OLLAMA_BASE_URL / OLLAMA_MODEL
```

## Run

```powershell
rag-api                            # or: uvicorn rag_chatbot.api.main:app --port 8000
```

Open http://127.0.0.1:8000/ , upload PDFs in the sidebar (progress is shown per document), then ask questions.
API docs: http://127.0.0.1:8000/docs

## API (`/api/v1`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Ollama reachability, models, indexed chunk count |
| POST | `/documents` | Upload PDFs (multipart `files`); ingestion runs in the background |
| GET | `/documents`, `/documents/{id}` | Ingestion status and progress |
| DELETE | `/documents/{id}` | Remove PDF and its vectors |
| POST | `/search` | Retrieval only: top-K passages with scores and provenance |
| POST | `/chat` | RAG answer + sources (file, pages) + citation check |
| POST | `/chat/stream` | Same, streamed as NDJSON (`sources`, `token`, `done`) |
| GET | `/llm/models` | Models on the Ollama endpoint |
| POST | `/llm/generate` | Plain completion through Ollama (no retrieval) |

## Latency tuning (target 2-5 s)

- Use a small instruct model in Ollama (e.g. `llama3.2:3b`, `qwen2.5:3b`) and keep it loaded (`OLLAMA_KEEP_ALIVE`).
- Use `/chat/stream`: first tokens arrive in about retrieval time + model time-to-first-token.
- `RERANKER_ENABLED=false` or a smaller `RETRIEVE_CANDIDATES` saves the most on CPU; a GPU makes both cheap.
- Stage timings (embed, search, rerank, LLM) are logged for every answer in the API console.
- Qdrant local mode searches by exact scan, which is fast for tens of thousands of chunks. For much larger corpora run a
  Qdrant server and set `QDRANT_URL` to get real HNSW search.

## Notes

- Identical PDFs are de-duplicated by content hash; chunk ids are deterministic, so re-ingestion is idempotent.
- Only one Qdrant local store can be open at a time: run a single API process.
- Ingestion is one document at a time (OCR is CPU/GPU heavy); queries stay available meanwhile but will be slower.
