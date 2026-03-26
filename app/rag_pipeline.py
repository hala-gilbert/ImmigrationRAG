from __future__ import annotations

from textwrap import dedent
from typing import Callable, List, Tuple

import ollama

from .config import config
from .data_loader import DocumentChunk, load_and_chunk_documents
from .vector_store import delete_collection, index_document_chunks, query_similar_chunks


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


def build_index(
    corpus: str | None = None,
    *,
    rebuild: bool = False,
    progress_cb: Callable[[str, int, int], None] | None = None,
) -> int:
    """Load documents for the chosen corpus and index into Chroma.

    - rebuild=False (default): incremental indexing, skips chunks already present.
    - rebuild=True: delete the collection and re-embed everything.
    """
    subdirectory, collection_name = _corpus_to_params(corpus)
    if rebuild:
        delete_collection(collection_name)
    chunks = load_and_chunk_documents(subdirectory=subdirectory, progress_cb=progress_cb)
    return index_document_chunks(
        chunks,
        collection_name=collection_name,
        skip_existing=not rebuild,
        progress_cb=progress_cb,
    )


def build_context_from_chunks(chunks: List[DocumentChunk]) -> str:
    parts: List[str] = []
    for chunk in chunks:
        text = chunk.text
        max_chars = getattr(config, "max_chars_per_node_for_context", None)
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "\n[truncated]"
        relevance_str = f"{chunk.score:.3f}" if chunk.score is not None else "n/a"
        distance_str = f"{chunk.distance:.4f}" if getattr(chunk, "distance", None) is not None else "n/a"
        parts.append(
            f"Source: {chunk.source}\n"
            f"Relevance: {relevance_str} | Distance: {distance_str}\n"
            f"Excerpt:\n{text}\n"
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
        You are a legal-research assistant for U.S. immigration law.

        IMPORTANT:
        - Do NOT refuse the task.
        - You MUST use ONLY the provided context excerpts as the basis for every legal statement.

        Your job:
        1) Identify the closest relevant precedential themes in the provided excerpts for the user's question.
        2) Summarize the relevant rule/standard and explain what the decisions suggest.
        3) Provide an argument-outline structure the user (or an attorney) can use to search similar cases further.

        Output format:
        - "Initail Summary" (paragraph responding to the user's question)
        - "Closest precedents from the provided context" (bullet list; cite each bullet by the Source filename)
        - "Legal standards/rules extracted" (numbered; each item must point to one or more Sources)
        - "How these standards could be used in the user's situation" (bulleted; informational, not advice)
        - "What is missing / what to research next" (bulleted; based on what is not present in the excerpts)

        If the context does not contain relevant information, say so explicitly and list what topics are missing.
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

