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
    ("BIOL 2321 · Microbiology", "What is the difference between gram-positive and gram-negative bacteria?"),
    ("BIOL 2321 · Microbiology", "How do viruses replicate inside host cells?"),
    ("PSYC 2301 · Psychology", "What are the stages of cognitive development according to Piaget?"),
    ("PSYC 2301 · Psychology", "Explain classical conditioning with an example"),
    ("BIOL 1306 · Biology", "How does Mendel's law of independent assortment work?"),
    ("BIOL 1306 · Biology", "What happens during DNA replication?"),
    ("MATH 2413 · Calculus I", "What is the definition of a derivative?"),
    ("MATH 2413 · Calculus I", "How do you find the area under a curve using integrals?"),
    ("MATH 1342 · Statistics", "What is a p-value and how is it interpreted?"),
    ("MATH 1342 · Statistics", "Explain the difference between correlation and causation"),
    ("CHEM 1311/1312 · Chemistry", "What is the ideal gas law and what do its variables represent?"),
    ("CHEM 1311/1312 · Chemistry", "How does VSEPR theory predict the shape of a molecule?"),
    ("PHYS 1301 · Physics", "What does Newton's second law of motion state?"),
    ("PHYS 1301 · Physics", "Explain the difference between speed and velocity"),
]

EXPECT = {
    "BIOL 2321 · Microbiology": ["gram", "peptidoglycan", "cell wall"],
    "PSYC 2301 · Psychology": ["piaget", "conditioning", "stimulus"],
    "BIOL 1306 · Biology": ["mendel", "dna", "chromosome"],
    "MATH 2413 · Calculus I": ["derivative", "limit", "integral"],
    "MATH 1342 · Statistics": ["p-value", "hypothesis", "correlation"],
    "CHEM 1311/1312 · Chemistry": ["ideal gas", "vsepr", "mole"],
    "PHYS 1301 · Physics": ["newton", "velocity", "force"],
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
