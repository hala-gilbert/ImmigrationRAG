from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv


load_dotenv()


class AppConfig(BaseModel):
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = Field(default_factory=lambda: Path("data") / "immigration")
    vector_db_dir: Path = Field(default_factory=lambda: Path(".chroma"))

    ollama_llm_model: str = Field(default_factory=lambda: os.getenv("OLLAMA_LLM_MODEL", "llama3.1"))
    ollama_embed_model: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    )

    # Larger defaults reduce chunk count significantly (faster indexing).
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "2500")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100")))
    top_k: int = Field(default_factory=lambda: int(os.getenv("TOP_K", "5")))

    @property
    def resolved_data_dir(self) -> Path:
        return (self.base_dir / self.data_dir).resolve()

    @property
    def resolved_vector_db_dir(self) -> Path:
        return (self.base_dir / self.vector_db_dir).resolve()


config = AppConfig()

