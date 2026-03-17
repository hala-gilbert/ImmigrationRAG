from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from rich.console import Console

from .config import config


console = Console()


@dataclass
class DocumentChunk:
    id: str
    text: str
    source: str


def iter_text_files(data_dir: Path | None = None) -> Iterable[Path]:
    """Yield all .txt files under the data directory.

    You can extend this to support PDF/HTML later.
    """
    root = data_dir or config.resolved_data_dir
    if not root.exists():
        console.print(f"[yellow]Data directory does not exist yet: {root}[/yellow]")
        return

    for path in root.rglob("*.txt"):
        if path.is_file():
            yield path


def _simple_chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = max(0, end - chunk_overlap)

    return chunks


def load_and_chunk_documents(subdirectory: str | None = None) -> List[DocumentChunk]:
    """Load immigration documents and return a list of chunks.

    If subdirectory is provided, documents are loaded from
    `config.resolved_data_dir / subdirectory`. This lets you keep
    separate corpora (e.g. removal, family, visas).
    """
    chunks: List[DocumentChunk] = []
    base_dir = config.resolved_data_dir
    data_dir = base_dir / subdirectory if subdirectory else base_dir
    chunk_size = config.chunk_size
    chunk_overlap = config.chunk_overlap

    console.print(f"[bold]Loading documents from[/bold] {data_dir}")

    for file_path in iter_text_files(data_dir):
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Failed to read {file_path}: {exc}[/red]")
            continue

        file_chunks = _simple_chunk_text(text, chunk_size, chunk_overlap)
        for idx, chunk_text in enumerate(file_chunks):
            chunk_id = f"{file_path.relative_to(data_dir)}::chunk-{idx}"
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    text=chunk_text,
                    source=str(file_path.relative_to(data_dir)),
                )
            )

    console.print(f"[green]Loaded {len(chunks)} chunks from documents.[/green]")
    return chunks

