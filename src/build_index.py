"""Build the GradPath textbook knowledge base.

Reads OpenStax textbook PDFs, extracts + cleans text, chunks it,
embeds with sentence-transformers, and stores a FAISS index plus
chunk metadata (book, chapter, pages) for citation.

Usage:
    python src/build_index.py
"""
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"

BOOKS = [
    # (pdf filename, display name, course label)
    ("microbiology.pdf", "Microbiology (OpenStax)", "BIOL 2321 · Microbiology"),
    ("psychology-2e.pdf", "Psychology 2e (OpenStax)", "PSYC 2301 · Psychology"),
    ("biology-2e.pdf", "Biology 2e (OpenStax)", "BIOL 1306 · Biology"),
    ("Calculus_Volume_1.pdf", "Calculus Volume 1 (OpenStax)", "MATH 2413 · Calculus I"),
    ("openstax-introductory-statistics.pdf", "Introductory Statistics (OpenStax)", "MATH 1342 · Statistics"),
    ("chemistry-2e.pdf", "Chemistry 2e (OpenStax)", "CHEM 1311/1312 · Chemistry"),
    ("college-physics.pdf", "College Physics (OpenStax)", "PHYS 1301 · Physics"),
]

CHUNK_WORDS = 450
OVERLAP_WORDS = 60
CHAPTER_RE = re.compile(r"^\s*chapter\s+(\d+)\b[:.\s]*(.*)$", re.IGNORECASE)


def clean_text(t: str) -> str:
    t = t.replace("-\n", "")          # de-hyphenate line breaks
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def extract_pages(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            txt = clean_text(page.extract_text() or "")
        except Exception:
            txt = ""
        # skip near-empty / front-matter pages
        if len(txt.split()) > 40:
            pages.append((i + 1, txt))
    return pages


def detect_chapter(text: str, current: str) -> str:
    head = " ".join(text.split()[:12])
    m = CHAPTER_RE.match(head)
    if m:
        title = m.group(2).strip()[:60]
        return f"Chapter {m.group(1)}" + (f": {title}" if title else "")
    return current


def chunk_book(pdf_file: str, book_name: str, course: str):
    pages = extract_pages(DATA / pdf_file)
    print(f"  {book_name}: {len(pages)} content pages")
    chunks, chapter = [], "Front matter"
    buf_words, buf_pages = [], []
    for pageno, text in pages:
        chapter = detect_chapter(text, chapter)
        words = text.split()
        # drop likely headers/footers repeated junk: very short all-caps lines already merged; keep simple
        buf_words.extend(words)
        buf_pages.append(pageno)
        while len(buf_words) >= CHUNK_WORDS:
            chunk_words = buf_words[:CHUNK_WORDS]
            chunks.append({
                "book": book_name, "course": course, "chapter": chapter,
                "page_start": buf_pages[0], "page_end": buf_pages[-1],
                "text": " ".join(chunk_words),
            })
            buf_words = buf_words[CHUNK_WORDS - OVERLAP_WORDS:]
            buf_pages = buf_pages[-2:] if len(buf_pages) > 2 else buf_pages
    if len(buf_words) > 120:
        chunks.append({
            "book": book_name, "course": course, "chapter": chapter,
            "page_start": buf_pages[0], "page_end": buf_pages[-1],
            "text": " ".join(buf_words),
        })
    return chunks


def main():
    ART.mkdir(exist_ok=True)
    all_chunks = []
    for pdf_file, book_name, course in BOOKS:
        p = DATA / pdf_file
        if not p.exists():
            print(f"  SKIP (missing): {pdf_file}")
            continue
        print(f"Chunking {book_name} ...")
        all_chunks.extend(chunk_book(pdf_file, book_name, course))
    print(f"Total chunks: {len(all_chunks)}")
    df = pd.DataFrame(all_chunks)
    df.to_parquet(ART / "chunks.parquet", index=False)

    print("Embedding chunks (all-MiniLM-L6-v2) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(df["text"].tolist(), batch_size=64,
                       show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(ART / "kb.index"))
    print(f"Saved: {ART/'chunks.parquet'} ({len(df)} chunks), {ART/'kb.index'}")
    print(df.groupby("course").size().to_string())


if __name__ == "__main__":
    main()
