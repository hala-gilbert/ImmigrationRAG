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

    st.title("Immigration RAG (Local Ollama)")
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
                with st.spinner("Indexing new documents/chunks..."):
                    count = build_index(selected_corpus, rebuild=False)
                st.success(f"Indexed {count} NEW chunks for corpus: {corpus_label}.")

        with col_b:
            if st.button("Force rebuild (re-embed all)", type="secondary"):
                with st.spinner("Deleting collection and rebuilding from scratch..."):
                    count = build_index(selected_corpus, rebuild=True)
                st.success(f"Rebuilt index with {count} chunks for corpus: {corpus_label}.")

        st.markdown("---")
        st.caption(
            "Place `.txt` immigration-law documents under "
            f"`{config.resolved_data_dir}`.\n"
            "For specific corpora, create subfolders like "
            "`removal`, `family`, or `visas` inside that directory.\n"
            "You can add PDF/other formats later with a custom loader."
        )

    st.subheader("Ask a question")
    user_question = st.text_area(
        "Your question",
        placeholder="Example: What are the general requirements for a family-based immigrant visa?",
        height=120,
    )

    if st.button("Get answer") and user_question.strip():
        with st.spinner("Thinking with RAG..."):
            result = answer_question(user_question.strip(), corpus=selected_corpus)

        answer_text = result["answer"]
        retrieved_chunks = result["retrieved_chunks"]

        st.markdown("### Answer")
        st.write(answer_text)

        with st.expander("Show retrieved context (top 3 by relevance)"):
            top_chunks = retrieved_chunks[:3]
            for idx, chunk in enumerate(top_chunks, start=1):
                score_display = f"{chunk.score:.3f}" if getattr(chunk, "score", None) is not None else "n/a"
                st.markdown(f"**Node {idx}**  |  **Source:** `{chunk.source}`  |  **Relevance:** {score_display}")
                st.code(chunk.text)
                st.markdown("---")


if __name__ == "__main__":
    main()

