"""Download only our curated official NBA pages; never crawl arbitrary URLs."""
import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .config import DATA, SOURCES


def extract_article(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # NBA's article body excludes navigation, related stories and footer text.
    article = soup.select_one('[class*="Article_article__body"]') or soup.find("article")
    if article is None:
        raise ValueError("NBA article body was not found; refusing to index navigation text.")
    for node in article.select("script, style, nav, footer, aside, button"):
        node.decompose()
    lines = []
    for node in article.select("h1, h2, h3, p, li, tr"):
        if node.find_parent(["li", "tr"]):
            continue
        line = " ".join(node.get_text(" ", strip=True).split())
        if line and (not lines or line != lines[-1]):
            lines.append(line)
    text = "\n".join(lines)
    if len(text) < 300:
        raise ValueError("Article was empty or unexpectedly short.")
    return text


def chunk_document(doc: dict, max_chars: int = 700, overlap: int = 1) -> list[dict]:
    """Preserve award/season rows, repeat the page title on every chunk."""
    rows = doc["text"].splitlines()
    season_rows = [line for line in rows if re.match(r"^(?:19|20)\d{2}[-–]\d{2}\b", line)]
    if season_rows:
        introduction = "\n".join(line for line in rows if line not in season_rows)
        records = ([introduction] if introduction else []) + season_rows
        return [{"id": f'{doc["url"]}#{i+1}', "title": doc["title"],
                 "url": doc["url"], "fetched_at": doc["fetched_at"],
                 "text": doc["title"] + "\n" + record}
                for i, record in enumerate(records)]
    chunks, lines, size = [], [], 0
    def emit():
        if lines:
            chunks.append({"id": f'{doc["url"]}#{len(chunks)+1}',
                           "title": doc["title"], "url": doc["url"],
                           "fetched_at": doc["fetched_at"],
                           "text": doc["title"] + "\n" + "\n".join(lines)})
    for line in doc["text"].splitlines():
        # Very long paragraphs are split too; award rows normally stay intact.
        for start in range(0, len(line), max_chars):
            part = line[start:start + max_chars]
            if lines and size + len(part) > max_chars:
                emit()
                lines = lines[-overlap:] if overlap else []
                size = sum(len(x) for x in lines)
            lines.append(part)
            size += len(part)
    emit()
    return chunks


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def ingest():
    documents, errors = [], []
    for title, url in SOURCES:
        try:
            response = requests.get(url, timeout=45, headers={"User-Agent": "NBA-Local-RAG/0.1 (personal educational project)"})
            response.raise_for_status()
            documents.append({"title": title, "url": url,
                              "fetched_at": datetime.now(timezone.utc).isoformat(),
                              "text": extract_article(response.text)})
            print(f"Downloaded {title}: {len(documents[-1]['text'])} characters", flush=True)
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"{title}: {exc}")
    # Never replace a healthy corpus with an incomplete refresh.
    if errors:
        raise RuntimeError("Refresh failed; existing documents kept.\n" + "\n".join(errors))
    atomic_json(DATA / "documents.json", documents)
    return documents


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    ingest()
