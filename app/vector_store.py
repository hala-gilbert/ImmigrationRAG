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

def delete_collection(collection_name: str = "immigration_rag") -> None:
    """Delete a collection from the persistent Chroma DB (destructive)."""
    client = get_chroma_client()
    try:
        client.delete_collection(name=collection_name)
    except Exception:  # noqa: BLE001
        # If it doesn't exist, treat as no-op.
        return


def _batched(seq: Sequence, batch_size: int):
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def index_document_chunks(
    chunks: Sequence[DocumentChunk],
    collection_name: str = "immigration_rag",
    batch_size: int = 1000,
    skip_existing: bool = True,
) -> int:
    """Index document chunks into Chroma and return the number indexed."""
    if not chunks:
        return 0

    collection = get_or_create_collection(collection_name)

    ids = [chunk.id for chunk in chunks]
    texts = [chunk.text for chunk in chunks]
    metadatas = [{"source": chunk.source} for chunk in chunks]

    # Chroma enforces a maximum batch size internally; keep our batches small and safe.
    total = 0
    for id_batch, text_batch, meta_batch in zip(
        _batched(ids, batch_size),
        _batched(texts, batch_size),
        _batched(metadatas, batch_size),
        strict=False,
    ):
        id_batch_list = list(id_batch)
        text_batch_list = list(text_batch)
        meta_batch_list = list(meta_batch)

        if skip_existing:
            # Only insert IDs that are not already present in the collection.
            existing = collection.get(ids=id_batch_list, include=[])
            existing_ids = set(existing.get("ids", []) or [])

            if existing_ids:
                filtered = [
                    (i, t, m)
                    for (i, t, m) in zip(id_batch_list, text_batch_list, meta_batch_list, strict=False)
                    if i not in existing_ids
                ]
                if not filtered:
                    continue
                id_batch_list, text_batch_list, meta_batch_list = map(list, zip(*filtered, strict=True))

        collection.upsert(ids=id_batch_list, documents=text_batch_list, metadatas=meta_batch_list)
        total += len(id_batch_list)

    return total


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
        include=["documents", "metadatas", "ids", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks: List[DocumentChunk] = []
    for doc_id, text, meta, dist in zip(ids, documents, metadatas, distances, strict=False):
        # Chroma returns a distance (smaller is better). Convert to a
        # similarity-style score between 0 and 1 where possible.
        score: float | None
        try:
            score = max(0.0, min(1.0, 1.0 - float(dist)))
        except Exception:  # noqa: BLE001
            score = None

        chunks.append(
            DocumentChunk(
                id=doc_id,
                text=text,
                source=meta.get("source", "unknown"),
                score=score,
            )
        )

    return chunks

