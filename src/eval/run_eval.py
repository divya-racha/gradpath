"""Offline retrieval evaluation harness for GradPath.

Retrieves top-K chunks per eval question using src/rag.py's TextbookRetriever
(NO LLM calls) and scores them against the expected chapter labels in
eval_set.jsonl. Writes a Markdown metrics table to artifacts/eval_report.md
and prints it to stdout.

Usage (from repo root):
    .venv/bin/python src/eval/run_eval.py

Metric definitions (K=5):
- precision@5 = (# relevant chunks in top-5) / 5
- recall@5    = (# relevant chunks in top-5) / (total chunks in the corpus whose
  chapter matches the question's expected chapter, restricted to the question's
  course). The chapter->count map is built ONCE per course up front, so the
  denominator is O(1) per question. If the corpus contains ZERO chunks labeled
  with the expected chapter, recall@5 is defined as 0.0 (unmeasurable) rather
  than undefined.
- MRR = 1 / (rank of the first relevant chunk), 0.0 if no relevant chunk.
- keyword_coverage@5 = fraction of expected_chapter_keywords that appear as
  case-insensitive substrings in the concatenated text of the top-5 chunks.
  This is a heuristic sanity signal independent of chapter labels.

A retrieved chunk counts as RELEVANT if its chapter matches the question's
expected_chapter:
  1. PRIMARY: compare chapter NUMBERS extracted via regex /chapter\\s*(\\d+)/i
     (also tolerates a bare leading "N:" / "N." label). Robust to chapter title
     strings changing (e.g. the sibling workstream's PDF-outline relabeling).
  2. FALLBACK: if either side has no extractable number, check whether the
     expected chapter's TITLE (text after "Chapter N:") appears as a
     case-insensitive substring of the chunk's chapter label. If exactly one
     side has a number, there is no match ("Front matter" never matches).

Aggregation: per-subject averages over that subject's questions; overall
MACRO = mean of the per-subject means (every subject weighted equally);
overall MICRO = mean over all questions (every question weighted equally).
"""
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag import TextbookRetriever  # noqa: E402  (reuse; do not reimplement)

K = 5
EVAL_PATH = Path(__file__).resolve().parent / "eval_set.jsonl"
ART = Path(__file__).resolve().parent.parent.parent / "artifacts"

CHAPTER_RE = re.compile(r"chapter\s*(\d+)", re.IGNORECASE)
LEADING_NUM_RE = re.compile(r"^\s*(\d+)\s*[:.]")


def chapter_num(label: str):
    """Extract the chapter number from a chapter label, or None."""
    if not label:
        return None
    m = CHAPTER_RE.search(label)
    if m:
        return int(m.group(1))
    m = LEADING_NUM_RE.match(label)
    return int(m.group(1)) if m else None


def chapter_title(label: str) -> str:
    """Title portion of an expected_chapter string (after 'Chapter N:')."""
    m = CHAPTER_RE.search(label or "")
    if not m:
        return (label or "").strip()
    return label[m.end():].lstrip(": ").strip()


def chapters_match(expected: str, actual: str) -> bool:
    n_exp, n_act = chapter_num(expected), chapter_num(actual)
    if n_exp is not None and n_act is not None:
        return n_exp == n_act
    if n_exp is not None or n_act is not None:
        return False  # one side numbered, the other not ("Front matter" never matches)
    title = chapter_title(expected)
    return bool(title) and title.lower() in (actual or "").lower()


def build_chapter_stats(df):
    """Per course: Counter of chapter numbers + Counter of raw chapter labels.

    Built once so the recall@5 denominator (corpus-wide relevant count) is O(1)
    per question instead of a full corpus scan each time.
    """
    stats = {}
    for course, cdf in df.groupby("course"):
        nums, labels = Counter(), Counter()
        for ch in cdf["chapter"]:
            labels[ch] += 1
            n = chapter_num(ch)
            if n is not None:
                nums[n] += 1
        stats[course] = {"numbers": nums, "labels": labels}
    return stats


def corpus_relevant_count(stats, course: str, expected: str) -> int:
    s = stats.get(course)
    if s is None:
        return 0
    n_exp = chapter_num(expected)
    if n_exp is not None:
        return s["numbers"].get(n_exp, 0)
    title = chapter_title(expected).lower()
    if not title:
        return 0
    return sum(c for ch, c in s["labels"].items() if title in ch.lower())


def evaluate(retriever, stats, questions):
    per_q = []
    for q in questions:
        results = retriever.search(q["question"], q["course"], k=K)
        rel_flags = [chapters_match(q["expected_chapter"], r["chapter"]) for r in results]
        n_rel = sum(rel_flags)
        denom = len(results) or 1
        corpus_total = corpus_relevant_count(stats, q["course"], q["expected_chapter"])
        blob = " ".join(r["text"] for r in results).lower()
        kws = q["expected_chapter_keywords"]
        kw_hits = sum(1 for kw in kws if kw.lower() in blob)
        per_q.append({
            "course": q["course"],
            "precision@5": n_rel / denom,
            "recall@5": (n_rel / corpus_total) if corpus_total else 0.0,
            "mrr": next((1.0 / (i + 1) for i, f in enumerate(rel_flags) if f), 0.0),
            "keyword_coverage@5": kw_hits / len(kws) if kws else 0.0,
        })
    return per_q


METRICS = ["precision@5", "recall@5", "mrr", "keyword_coverage@5"]


def aggregate(per_q):
    by_course = defaultdict(list)
    for r in per_q:
        by_course[r["course"]].append(r)
    subj = {c: {m: sum(r[m] for r in rs) / len(rs) for m in METRICS}
            for c, rs in by_course.items()}
    macro = {m: sum(s[m] for s in subj.values()) / len(subj) for m in METRICS}
    micro = {m: sum(r[m] for r in per_q) / len(per_q) for m in METRICS}
    return subj, macro, micro, {c: len(rs) for c, rs in by_course.items()}


def fmt_row(cells, widths):
    return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"


def render_table(subj, macro, micro, counts):
    header = ["subject", "n"] + METRICS
    rows = []
    for course in sorted(subj):
        s = subj[course]
        rows.append([course, str(counts[course])] + [f"{s[m]:.3f}" for m in METRICS])
    rows.append(["**macro avg** (subjects equal)", str(sum(counts.values()))] +
                [f"**{macro[m]:.3f}**" for m in METRICS])
    rows.append(["**micro avg** (questions equal)", str(sum(counts.values()))] +
                [f"**{micro[m]:.3f}**" for m in METRICS])
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    lines = [fmt_row(header, widths),
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines += [fmt_row(r, widths) for r in rows]
    return "\n".join(lines)


def main():
    t0 = time.time()
    questions = [json.loads(l) for l in EVAL_PATH.read_text().splitlines() if l.strip()]
    print(f"Loading retriever (this downloads nothing; model is cached) ...")
    r = TextbookRetriever()
    stats = build_chapter_stats(r.df)
    print(f"Index: {len(r.df)} chunks, {len(r.courses)} courses "
          f"({time.time()-t0:.1f}s). Evaluating {len(questions)} questions ...")
    t1 = time.time()
    per_q = evaluate(r, stats, questions)
    print(f"Retrieval+scoring took {time.time()-t1:.1f}s\n")
    subj, macro, micro, counts = aggregate(per_q)
    table = render_table(subj, macro, micro, counts)

    report = (
        f"# GradPath retrieval eval report\n\n"
        f"- Date: {date.today().isoformat()}\n"
        f"- Questions: {len(questions)} ({', '.join(f'{c}: {counts[c]}' for c in sorted(counts))})\n"
        f"- Chunks in index: {len(r.df)}\n"
        f"- k = {K}; relevance = chapter-number match (fallback: title substring)\n"
        f"- recall@5 denominator = corpus chunks in the expected chapter (per course), "
        f"precomputed per course; 0.0 when the corpus has no chunk labeled with the expected chapter\n\n"
        f"{table}\n\n"
        f"## Notes\n\n"
        f"- Chapter labels come from PDF-outline extraction (per-book coverage "
        f"86.7-99.4%); remaining non-chapter labels are honest back-matter "
        f"(Preface, Glossary, Index, appendices).\n"
        f"- recall@5's denominator is EVERY corpus chunk in the expected chapter "
        f"(often hundreds), so it is structurally low by design -- treat "
        f"precision@5 / MRR / keyword_coverage@5 as the headline metrics.\n"
        f"- keyword_coverage@5 is label-independent: fraction of distinctive "
        f"expected-chapter phrases found in the top-5 chunks' text.\n"
    )
    ART.mkdir(exist_ok=True)
    (ART / "eval_report.md").write_text(report)
    print(table)
    print(f"\nWrote {ART / 'eval_report.md'}")


if __name__ == "__main__":
    main()
