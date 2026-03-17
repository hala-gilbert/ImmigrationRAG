from __future__ import annotations

from textwrap import dedent
from typing import List, Tuple

import ollama

from .config import config
from .data_loader import DocumentChunk, load_and_chunk_documents
from .vector_store import index_document_chunks, query_similar_chunks


def _corpus_to_params(corpus: str | None) -> Tuple[str | None, str]:
    """Map a high-level corpus name to data subdirectory and collection name."""
    key = (corpus or "").strip().lower()

    if key == "removal":
        return "removal", "immigration_removal"
    if key == "family":
        return "family", "immigration_family"
    if key == "visas":
        return "visas", "immigration_visas"

    # Default: use base data directory and default collection
    return None, "immigration_rag"


def build_index(corpus: str | None = None) -> int:
    """Load documents for the chosen corpus, chunk them, and index into Chroma."""
    subdirectory, collection_name = _corpus_to_params(corpus)
    chunks = load_and_chunk_documents(subdirectory=subdirectory)
    return index_document_chunks(chunks, collection_name=collection_name)


def build_context_from_chunks(chunks: List[DocumentChunk]) -> str:
    parts: List[str] = []
    for chunk in chunks:
        parts.append(
            f"Source: {chunk.source}\n"
            f"Excerpt:\n{chunk.text}\n"
            "-------------------------"
        )
    return "\n".join(parts)


def answer_question(question: str, corpus: str | None = None) -> dict:
    """Run a full RAG cycle and return the model's answer + context."""
    _, collection_name = _corpus_to_params(corpus)
    retrieved_chunks = query_similar_chunks(question, collection_name=collection_name)
    context = build_context_from_chunks(retrieved_chunks)

    system_prompt = dedent(
        """
        You are an assistant specialized in U.S. immigration law.
        Use ONLY the provided context excerpts from primary sources and authoritative
        guidance when answering. If the answer is unclear or not covered, say that
        you are not certain rather than guessing.

        When you quote or summarize, clearly indicate it is based on the supplied context.
        """
    ).strip()

    prompt = dedent(
        f"""
        System instructions:
        {system_prompt}

        Context:
        {context}

        User question:
        {question}
        """
    ).strip()

    response = ollama.chat(
        model=config.ollama_llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    answer_text = response["message"]["content"]

    return {
        "answer": answer_text,
        "retrieved_chunks": retrieved_chunks,
        "raw_response": response,
    }

