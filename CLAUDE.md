# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG chatbot API built with FastAPI, ChromaDB, and OpenAI. Designed to be embedded on a company website. It answers questions using retrieved company documents, captures customer leads in SQLite, and books meetings via Cal.com.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in env vars
cp .env.example .env

# Run the API server (from project root)
uvicorn app.main:app --reload

# Ingest documents into ChromaDB
python scripts/ingest_docs.py --dir data/documents
python scripts/ingest_docs.py --file path/to/file.pdf
```

## Architecture

```
app/
├── main.py               FastAPI app with lifespan (creates DB tables on startup), CORS
├── config.py             Pydantic Settings loaded from .env
├── api/
│   ├── chat.py           POST /api/v1/chat — main chat endpoint
│   └── admin.py          GET  /api/v1/admin/leads — list leads (X-Api-Key header required)
├── db/
│   ├── models.py         SQLAlchemy Lead model
│   └── database.py       SQLite engine, get_db dependency, create_tables()
├── rag/
│   ├── ingest.py         PDF/TXT → chunk → ChromaDB upsert (idempotent via SHA256 IDs)
│   └── retriever.py      ChromaDB query, singleton collection client
└── services/
    ├── chat_engine.py    Tool-calling loop: calls OpenAI, dispatches tools, loops until no tool_calls
    ├── session_store.py  In-memory sessions dict, 30-min TTL, 40-message cap
    └── calcom.py         Cal.com v2 API: get_available_slots, create_booking
scripts/
└── ingest_docs.py        CLI wrapper for rag/ingest.py
data/
└── documents/            Drop PDFs and TXT files here before ingesting
```

## Key Design Decisions

**Tool-calling loop** (`chat_engine.py:run_chat`): OpenAI may request multiple sequential tool calls. The loop continues calling OpenAI with accumulated tool results until a response has no `tool_calls`. Capped at 8 iterations. The 4 tools are: `search_knowledge_base`, `store_lead`, `get_available_slots`, `book_meeting`.

**Session history**: Stored in memory keyed by `session_id` (UUID). Sessions expire after 30 min idle. History is capped at 40 messages to stay within context limits. The `session_id` is returned on every response and must be sent back by the client on subsequent messages.

**ChromaDB**: Uses `OpenAIEmbeddingFunction` from `chromadb.utils.embedding_functions` for embeddings. The retriever uses a module-level singleton client to avoid reconnecting on every request. ChromaDB data is persisted at `CHROMA_PERSIST_DIR` (default `./data/chroma_store`).

**Ingestion is idempotent**: Chunk IDs are SHA256 hashes of `filename::chunk_index`, so re-ingesting the same file overwrites existing chunks instead of duplicating them.

**Cal.com**: Requires `CALCOM_API_KEY` and `CALCOM_EVENT_TYPE_ID` in `.env`. Uses cal-api-version `2024-08-13`. If not configured, tool calls return a graceful error string to the model.

**Admin auth**: The `GET /api/v1/admin/leads` endpoint requires `X-Api-Key: <ADMIN_API_KEY>` header.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `OPENAI_MODEL` | No | Default: `gpt-4o-mini` |
| `OPENAI_EMBEDDING_MODEL` | No | Default: `text-embedding-3-small` |
| `CALCOM_API_KEY` | For meetings | Cal.com API key |
| `CALCOM_EVENT_TYPE_ID` | For meetings | Cal.com event type ID |
| `ADMIN_API_KEY` | Yes | Key for the admin leads endpoint |
| `ALLOWED_ORIGINS` | No | Comma-separated CORS origins, default `*` |
| `COMPANY_NAME` | No | Injected into the system prompt |
