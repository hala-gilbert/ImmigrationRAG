from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.config import config
from app.rag_pipeline import answer_question, build_index


def ensure_data_dir() -> None:
    data_dir: Path = config.resolved_data_dir
    data_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    st.set_page_config(
        page_title="Immigration RAG - beta",
        page_icon="📚",
        layout="wide",
    )

    st.title("Immigration Legal RAG (Local Ollama)")
    st.write(
        "Ask questions about U.S. immigration law using a local Retrieval-Augmented Generation setup.\n"
        "Make sure Ollama is running and your immigration documents are placed in "
        f"`{config.resolved_data_dir}`."
    )

    ensure_data_dir()

    with st.sidebar:
        st.header("Corpus")
        corpus_options = {
            "Removal / Deportation": "removal",
            "Family / Green Cards": "family",
            "Visas": "visas",
            "All documents (default folder)": None,
        }
        corpus_label = st.selectbox(
            "Choose corpus",
            list(corpus_options.keys()),
            index=0,
        )
        selected_corpus = corpus_options[corpus_label]

        st.markdown("---")
        st.header("Index & Settings")
        st.markdown("**Ollama models**")
        st.text_input("LLM model", value=config.ollama_llm_model, disabled=True)
        st.text_input("Embedding model", value=config.ollama_embed_model, disabled=True)
        st.markdown("---")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Update index (skip existing)", type="primary"):
                progress = st.progress(0, text="Starting…")

                def _cb(phase: str, cur: int, total: int) -> None:
                    if total <= 0:
                        return
                    pct = max(0.0, min(1.0, cur / total))
                    label = "Working…"
                    if phase == "chunking_files":
                        label = f"Chunking files: {cur}/{total}"
                    elif phase == "indexing_batches":
                        label = f"Indexing batches: {cur}/{total}"
                    progress.progress(pct, text=label)

                with st.spinner("Indexing new documents/chunks..."):
                    count = build_index(selected_corpus, rebuild=False, progress_cb=_cb)
                st.success(f"Indexed {count} NEW chunks for corpus: {corpus_label}.")

        with col_b:
            if st.button("Force rebuild (re-embed all)", type="secondary"):
                progress = st.progress(0, text="Starting…")

                def _cb(phase: str, cur: int, total: int) -> None:
                    if total <= 0:
                        return
                    pct = max(0.0, min(1.0, cur / total))
                    label = "Working…"
                    if phase == "chunking_files":
                        label = f"Chunking files: {cur}/{total}"
                    elif phase == "indexing_batches":
                        label = f"Indexing batches: {cur}/{total}"
                    progress.progress(pct, text=label)

                with st.spinner("Deleting collection and rebuilding from scratch..."):
                    count = build_index(selected_corpus, rebuild=True, progress_cb=_cb)
                st.success(f"Rebuilt index with {count} chunks for corpus: {corpus_label}.")

        st.markdown("---")
        st.caption(
            "Place `.txt` immigration-law documents under "
            f"`{config.resolved_data_dir}`.\n"
            "For specific corpora, subfolders hae been created:"
            "`removal`, `family`, and `visas` (visas is the only one currently populated)"
        )

    st.subheader("Ask a question")
    user_question = st.text_area(
        "Your question",
        placeholder="Example: What are the general requirements for a family-based immigrant visa?",
        height=120,
    )

    if st.button("Get answer") and user_question.strip():
        with st.spinner("Thinking..."):
            result = answer_question(user_question.strip(), corpus=selected_corpus)

        answer_text = result["answer"]
        retrieved_chunks = result["retrieved_chunks"]

        st.markdown("### Answer")
        st.write(answer_text)

        with st.expander("Show retrieved context (top 3 by relevance)"):
            top_chunks = retrieved_chunks[:3]
            for idx, chunk in enumerate(top_chunks, start=1):
                relevance_display = f"{chunk.score:.3f}" if getattr(chunk, "score", None) is not None else "n/a"
                distance_display = f"{chunk.distance:.4f}" if getattr(chunk, "distance", None) is not None else "n/a"
                st.markdown(
                    f"**Chunk {idx}**  |  **Source:** `{chunk.source}`  \n"
                    f"**Relevance:** {relevance_display}  |  **Distance:** {distance_display}"
                    #f"**Excerpt:**\n{chunk.text}"
                )
                st.code(chunk.text)
                st.markdown("---")


if __name__ == "__main__":
    main()

