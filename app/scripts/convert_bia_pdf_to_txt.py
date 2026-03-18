from pathlib import Path

from pypdf import PdfReader

RAW_DIR = Path("data/immigration/visas/cases_bia_ag_raw")
OUT_DIR = Path("data/immigration/visas/cases_bia_ag")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def pdf_to_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)

def main() -> None:
    for pdf_path in RAW_DIR.glob("*.pdf"):
        out_path = OUT_DIR / (pdf_path.stem + ".txt")
        if out_path.exists():
            continue
        try:
            text = pdf_to_text(pdf_path)
            out_path.write_text(text, encoding="utf-8", errors="ignore")
            print(f"wrote {out_path}")
        except Exception as e:
            print(f"FAILED {pdf_path}: {e}")

if __name__ == "__main__":
    main()