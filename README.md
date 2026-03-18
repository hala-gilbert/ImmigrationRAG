# ImmigrationRAG

Local RAG (Retrieval-Augmented Generation) assistant for U.S. immigration law, using:

- **Ollama** for local LLM + embeddings
- **ChromaDB** for vector storage
- **Streamlit** for a simple GUI

## Prerequisites

- Python 3.10+ recommended
- [Ollama](https://ollama.com) installed and running locally
  - At least one LLM model pulled (e.g. `ollama pull llama3.1`)
  - At least one embedding model pulled (e.g. `ollama pull nomic-embed-text`)

## Setup

```bash
cd ImmigrationRAG
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (optional) to override defaults:

```bash
OLLAMA_LLM_MODEL=llama3.1
OLLAMA_EMBED_MODEL=nomic-embed-text
DATA_DIR=data/immigration
VECTOR_DB_DIR=.chroma
CHUNK_SIZE=2500
CHUNK_OVERLAP=100
TOP_K=5
```

## Running and using the app

Run ```ollama serve``` to start up a local ollama server (make sure ollama is locally installed first)

make sure llama3.1 and nomic-embed-text is downloaded. if not run:
```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

Then run:
```bash
streamlit run streamlit_app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## Project structure (initial)

- `streamlit_app.py` – Streamlit GUI entrypoint
- `app/config.py` – Configuration and environment variables
- `app/data_loader.py` – Load and chunk immigration documents (pdf/txt)
- `app/vector_store.py` – Build and query the Chroma vector store using Ollama embeddings
- `app/rag_pipeline.py` – High-level RAG pipeline (retrieve + generate via Ollama)

After loading in the data make sure to build/rebuild index before asking legal questions.

### RAG notes
- Uses Cosine similarity
- **distance** from each grabbed node is displayed. Ranges 0 - 1, 0 being closest to prompt, 1 being furtherst from the prompt.
-  **relevance** of each node grabbed is also displayed. Ranges 0 - 1, 0 meaning no relevance, 1 being identitcal.