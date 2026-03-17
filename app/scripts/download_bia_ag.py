import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Put downloads here (pdf/html)
OUT_DIR = Path("data/immigration/visas/cases_bia_ag_raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Option A: paste your metadata JSON into a file and point here:
METADATA_JSON_PATH = Path("bai_ag_bia_metadata.json")

# Be polite to DOJ servers
SLEEP_SECONDS = 0.5

session = requests.Session()
session.headers.update(
    {
        "User-Agent": "ImmigrationRAG-research-downloader (personal academic use)",
    }
)

def safe_filename(url: str) -> str:
    path = urlparse(url).path
    name = Path(path).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name or "download.bin"

def download_file(url: str) -> Path:
    out_path = OUT_DIR / safe_filename(url)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    r = session.get(url, stream=True, timeout=60)
    r.raise_for_status()

    with out_path.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)

    return out_path

def extract_decision_links(volume_url: str) -> set[str]:
    r = session.get(volume_url, timeout=60)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    links = set()

    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue

        abs_url = urljoin(volume_url, href)

        # Heuristic: take PDFs; you can broaden later if needed
        if abs_url.lower().endswith(".pdf"):
            links.add(abs_url)

    return links

def main() -> None:
    meta = json.loads(METADATA_JSON_PATH.read_text(encoding="utf-8"))
    volume_urls = [d["accessURL"] for d in meta.get("distribution", []) if "accessURL" in d]

    all_links: set[str] = set()
    for vurl in volume_urls:
        print(f"[volume] {vurl}")
        try:
            links = extract_decision_links(vurl)
            print(f"  found {len(links)} pdf links")
            all_links |= links
        except Exception as e:
            print(f"  ERROR: {e}")

        time.sleep(SLEEP_SECONDS)

    print(f"\nTotal unique PDF links: {len(all_links)}")
    for url in sorted(all_links):
        try:
            path = download_file(url)
            print(f"downloaded: {path}")
        except Exception as e:
            print(f"FAILED: {url} -> {e}")
        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    main()