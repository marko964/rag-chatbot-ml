"""Document ingestion: load PDFs/TXT files, chunk, and store in ChromaDB."""
import hashlib
from pathlib import Path
from typing import List

import fitz  # pymupdf
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from app.config import settings

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def get_collection(client: chromadb.PersistentClient = None):
    if client is None:
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.openai_embedding_model,
    )
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=embedding_fn,
    )


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 80) -> List[str]:
    """Split text into overlapping character-based chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _extract_pdf(path: Path) -> List[tuple[str, int]]:
    """Return list of (text, page_number) tuples."""
    doc = fitz.open(str(path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append((text, i + 1))
    return pages


def _extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _chunk_id(source: str, chunk_index: int) -> str:
    """Deterministic chunk ID for idempotent upserts."""
    raw = f"{source}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def ingest_file(path: Path, collection=None) -> int:
    """Ingest a single PDF or TXT file. Returns number of chunks stored."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    if collection is None:
        collection = get_collection()

    documents, ids, metadatas = [], [], []

    if path.suffix.lower() == ".pdf":
        for page_text, page_num in _extract_pdf(path):
            for i, chunk in enumerate(_chunk_text(page_text)):
                chunk_index = len(documents)
                documents.append(chunk)
                ids.append(_chunk_id(path.name, chunk_index))
                metadatas.append({"source": path.name, "page": page_num, "chunk": i})
    else:
        text = _extract_txt(path)
        for i, chunk in enumerate(_chunk_text(text)):
            documents.append(chunk)
            ids.append(_chunk_id(path.name, i))
            metadatas.append({"source": path.name, "chunk": i})

    if documents:
        collection.upsert(documents=documents, ids=ids, metadatas=metadatas)

    return len(documents)


def ingest_directory(directory: Path) -> dict[str, int]:
    """Ingest all supported files in a directory. Returns {filename: chunk_count}."""
    collection = get_collection()
    results = {}
    for path in sorted(Path(directory).iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            count = ingest_file(path, collection)
            results[path.name] = count
    return results
