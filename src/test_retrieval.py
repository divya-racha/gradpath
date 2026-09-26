"""Smoke-test the GradPath knowledge base: retrieval quality per course.

Usage: .venv/bin/python src/test_retrieval.py
No API key needed — tests retrieval only, not generation.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag import TextbookRetriever

QUERIES = [
    ("Microbiology", "What is the difference between gram-positive and gram-negative bacteria?"),
    ("Microbiology", "How do viruses replicate inside host cells?"),
    ("Psychology", "What are the stages of cognitive development according to Piaget?"),
    ("Psychology", "Explain classical conditioning with an example"),
    ("Biology / Genetics", "How does Mendel's law of independent assortment work?"),
    ("Biology / Genetics", "What happens during DNA replication?"),
    ("Calculus I", "What is the definition of a derivative?"),
    ("Calculus I", "How do you find the area under a curve using integrals?"),
    ("Statistics", "What is a p-value and how is it interpreted?"),
    ("Statistics", "Explain the difference between correlation and causation"),
]

EXPECT = {
    "Microbiology": ["gram", "peptidoglycan", "cell wall"],
    "Psychology": ["piaget", "conditioning", "stimulus"],
    "Biology / Genetics": ["mendel", "dna", "chromosome"],
    "Calculus I": ["derivative", "limit", "integral"],
    "Statistics": ["p-value", "hypothesis", "correlation"],
}


def main():
    t0 = time.time()
    r = TextbookRetriever()
    print(f"Loaded index in {time.time()-t0:.1f}s | courses: {r.courses}")
    print(f"Total chunks: {len(r.df)}\n")
    hits = 0
    for course, q in QUERIES:
        res = r.search(q, course, k=3)
        assert res, f"no results for {course}: {q}"
        top = res[0]
        kw = EXPECT[course]
        ok = any(k in top["text"].lower() for k in kw)
        hits += ok
        mark = "✅" if ok else "⚠️"
        print(f"{mark} [{course}] {q[:60]}")
        print(f"    → {TextbookRetriever.cite(top)} (score {top['score']:.3f})")
    print(f"\n{hits}/{len(QUERIES)} queries returned keyword-relevant top chunks")


if __name__ == "__main__":
    main()
