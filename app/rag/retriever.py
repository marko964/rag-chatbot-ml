"""Query ChromaDB for relevant document chunks."""
from typing import List

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from app.config import settings

_client: chromadb.PersistentClient = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        embedding_fn = OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key,
            model_name=settings.openai_embedding_model,
        )
        _collection = _client.get_or_create_collection(
            name=settings.chroma_collection_name,
            embedding_function=embedding_fn,
        )
    return _collection


def retrieve(query: str, n_results: int = 5) -> List[str]:
    """Return top-N relevant document chunks for a query."""
    collection = _get_collection()
    try:
        results = collection.query(query_texts=[query], n_results=n_results)
        return results.get("documents", [[]])[0]
    except Exception:
        return []
