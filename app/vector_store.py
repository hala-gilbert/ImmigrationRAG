from __future__ import annotations

from typing import List, Sequence

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction
import ollama

from .config import config
from .data_loader import DocumentChunk


class OllamaEmbeddingFunction(EmbeddingFunction):
    """Chroma-compatible embedding function backed by Ollama."""

    def __call__(self, texts: Sequence[str]) -> List[List[float]]:  # type: ignore[override]
        embeddings: List[List[float]] = []
        for text in texts:
            response = ollama.embeddings(
                model=config.ollama_embed_model,
                prompt=text,
            )
            embeddings.append(response["embedding"])
        return embeddings


def get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=str(config.resolved_vector_db_dir),
        settings=Settings(allow_reset=False),
    )


def get_or_create_collection(collection_name: str = "immigration_rag"):
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=OllamaEmbeddingFunction(),
    )


def index_document_chunks(chunks: Sequence[DocumentChunk], collection_name: str = "immigration_rag") -> int:
    """Index document chunks into Chroma and return the number indexed."""
    if not chunks:
        return 0

    collection = get_or_create_collection(collection_name)

    ids = [chunk.id for chunk in chunks]
    texts = [chunk.text for chunk in chunks]
    metadatas = [{"source": chunk.source} for chunk in chunks]

    collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    return len(ids)


def query_similar_chunks(
    query: str,
    top_k: int | None = None,
    collection_name: str = "immigration_rag",
) -> List[DocumentChunk]:
    """Retrieve the top_k most similar chunks to the query."""
    k = top_k or config.top_k
    collection = get_or_create_collection(collection_name)

    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "ids"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]

    chunks: List[DocumentChunk] = []
    for doc_id, text, meta in zip(ids, documents, metadatas, strict=False):
        chunks.append(
            DocumentChunk(
                id=doc_id,
                text=text,
                source=meta.get("source", "unknown"),
            )
        )

    return chunks

