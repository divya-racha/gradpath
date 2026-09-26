"""Retrieval over the GradPath textbook knowledge base."""
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"


class TextbookRetriever:
    def __init__(self):
        self.df = pd.read_parquet(ART / "chunks.parquet")
        self.index = faiss.read_index(str(ART / "kb.index"))
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    @property
    def courses(self):
        return sorted(self.df["course"].unique().tolist())

    def search(self, query: str, course: str, k: int = 5):
        """Return top-k chunks for a course: list of dicts with text + citation."""
        q = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        course_mask = (self.df["course"] == course).to_numpy()
        idx = np.where(course_mask)[0]
        if len(idx) == 0:
            return []
        sub = faiss.IndexFlatIP(self.index.d)
        sub.add(self.index.reconstruct_batch(idx))
        scores, pos = sub.search(q, min(k, len(idx)))
        out = []
        for s, p in zip(scores[0], pos[0]):
            row = self.df.iloc[idx[p]]
            out.append({
                "text": row["text"],
                "book": row["book"],
                "chapter": row["chapter"],
                "page_start": int(row["page_start"]),
                "page_end": int(row["page_end"]),
                "score": float(s),
            })
        return out

    @staticmethod
    def cite(c) -> str:
        pages = f"p. {c['page_start']}" if c["page_start"] == c["page_end"] \
            else f"pp. {c['page_start']}–{c['page_end']}"
        chap = "" if c["chapter"] == "Front matter" else f", {c['chapter']}"
        return f"{c['book']}{chap}, {pages}"
