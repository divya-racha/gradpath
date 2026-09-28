"""Build the GradPath textbook knowledge base.

Reads OpenStax textbook PDFs, extracts + cleans text, chunks it,
embeds with sentence-transformers, and stores a FAISS index plus
chunk metadata (book, chapter, pages) for citation.

Chapter detection (v2): each PDF's outline/bookmarks are parsed with
pypdf to map chapter titles -> page ranges, and every chunk is labeled
with the chapter (and section, when known) containing its pages, e.g.
"Chapter 3: The Cellular Level -- 3.2 Comparing Prokaryotic and
Eukaryotic Cells". Books whose outline destinations are broken fall
back to text-based detection (chapter-intro page markers for the 2e
books, chapter-opener page heads for College Physics). The old
"Chapter N" page-head regex remains as a last-resort fallback.

Usage:
    python src/build_index.py
"""
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from pypdf import PdfReader
from pypdf.generic import IndirectObject
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
# Last-resort fallback: page head starts with "Chapter N ..."
CHAPTER_RE = re.compile(r"^\s*chapter\s+(\d+)\b[:.\s]*(.*)$", re.IGNORECASE)
# Chapter number embedded in an outline title, e.g. "Chapter 1 Essential Ideas"
OUTLINE_CHAPTER_RE = re.compile(r"(?i)^chapter\s*(\d+)[.\s:\-–]*(.*)$")
APPENDIX_RE = re.compile(r"(?i)^appendix\s+([A-Z])[\s.\-–]*(.*)$")
# College Physics chapter openers, e.g. "1 INTRODUCTION: THE NATURE OF SCIENCE AND PHYSICS"
PHYS_CHAPTER_RE = re.compile(
    r"^(\d+)\s+([A-Z][A-Z\s:,&'\-–()]{4,80}?)(?=\s+(Figure|By the end|$))")
# OpenStax 2e chapter splash pages start with this marker
INTRO_MARKER = 'Introduction class="introduction"'
# Back-matter headings detectable from page heads
BACKMATTER_RE = re.compile(r"(?i)^(answer key|glossary|index|references|appendix)\b")
FRONTBACK_RE = re.compile(
    r"(?i)^(preface|contents|index|glossary|answer key|references|solutions|"
    r"review exercises|practice tests|data sets|appendix)")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def clean_title(s: str) -> str:
    """Normalize an outline/heading title for display."""
    return norm(s).rstrip("*").strip()


def smart_title(s: str) -> str:
    """Title-case an ALL-CAPS heading without mangling apostrophes."""
    return " ".join(w[:1].upper() + w[1:].lower() if w else w for w in s.split())


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


def page_heads(reader):
    """Cleaned full text of every page (0-based list)."""
    heads = []
    for pg in reader.pages:
        try:
            t = clean_text(pg.extract_text() or "")
        except Exception:
            t = ""
        heads.append(t)
    return heads


# ----------------------------------------------------------------------------
# Outline helpers
# ----------------------------------------------------------------------------

def ref_to_index(reader):
    ref2idx = {}
    for i, pg in enumerate(reader.pages):
        ref = pg.indirect_reference
        if ref is not None:
            ref2idx[(ref.idnum, ref.generation)] = i
    return ref2idx


def dest_page_index(ref2idx, item):
    """0-based PDF page index for an outline item, or None if unresolvable."""
    try:
        arr = item.dest_array
    except AttributeError:
        return None
    if not arr:
        return None
    pgref = arr[0]
    if isinstance(pgref, IndirectObject):
        return ref2idx.get((pgref.idnum, pgref.generation))
    return None


def flatten_outline(outline):
    out = []

    def walk(ol, d):
        for it in ol:
            if isinstance(it, list):
                walk(it, d + 1)
            else:
                out.append((d, it))

    walk(outline, 0)
    return out


def outline_entries(reader):
    """[(depth, title, page0|None)] for the whole outline; [] if unreadable."""
    try:
        ol = reader.outline
    except Exception:
        return []
    ref2idx = ref_to_index(reader)
    entries = []
    first = True
    for d, it in flatten_outline(ol):
        title = norm(str(it.title))
        p0 = dest_page_index(ref2idx, it)
        # Guard against destinations that collapsed to page 0 (broken links):
        # a real target is never the very first PDF page for non-first items.
        if p0 == 0 and not first:
            p0 = None
        first = False
        entries.append((d, title, p0))
    return entries


# ----------------------------------------------------------------------------
# Chapter-map builders: each returns
#   blocks   = [(start0, label, is_chapter, chap_num|None)]
#   sections = [(page0, chap_num, section_title)]
# covering the book in page order; block end = next block start.
# ----------------------------------------------------------------------------

def build_from_outline(entries, npages, heads):
    """Chapter map from outline destinations (chemistry, calculus, biology)."""
    if not entries:
        return None
    max_depth = max(d for d, _, _ in entries)
    cdepth = 1 if max_depth >= 2 else 0  # units sit at depth 0 when nested deeper
    chap_items = [(t, p) for d, t, p in entries if d == cdepth]
    numbered = any(OUTLINE_CHAPTER_RE.match(t) for t, _ in chap_items)

    blocks, sections = [], []
    chap_num = 0
    chapters = []  # (num, title, start0)
    for t, p in chap_items:
        m = OUTLINE_CHAPTER_RE.match(t)
        if m:
            chap_num = int(m.group(1))
            ctitle = m.group(2).strip() or t
            chapters.append((chap_num, ctitle, p))
        elif numbered:
            # non-chapter item at chapter depth (Contents, Preface, Appendix..)
            am = APPENDIX_RE.match(t)
            label = f"Appendix {am.group(1)}: {am.group(2).strip()}" if am else t
            if p is not None:
                blocks.append((p, label, False, None))
        elif FRONTBACK_RE.match(t):
            if p is not None:
                blocks.append((p, t, False, None))
        else:
            chap_num += 1
            chapters.append((chap_num, t, p))

    # Drop chapters whose destination could not be resolved; they get merged
    # into the previous block (diagnostic printed by caller).
    good_chapters = [c for c in chapters if c[2] is not None]
    for num, ctitle, p in good_chapters:
        blocks.append((p, f"Chapter {num}: {ctitle}", True, num))

    # Sections: outline items one level below the chapter depth.
    sec_items = [(t, p) for d, t, p in entries if d == cdepth + 1
                 and p is not None and t and not OUTLINE_CHAPTER_RE.match(t)
                 and not FRONTBACK_RE.match(t)]
    # assign each section to the chapter whose page range contains it
    starts = sorted(b[0] for b in blocks)
    for t, p in sec_items:
        owner = None
        for s in starts:
            if s <= p:
                owner = s
        if owner is None:
            continue
        blk = next(b for b in blocks if b[0] == owner)
        if blk[2]:  # owner is a chapter
            sections.append((p, blk[3], clean_title(t)))

    # Trailing unit-level items after the last chapter (e.g. Biology 2e's
    # appendix tables at depth 0 while chapters sit at depth 1).
    if cdepth > 0:
        last_ch_start = max((b[0] for b in blocks if b[2]), default=0)
        for d, t, p in entries:
            if d < cdepth and p is not None and p >= last_ch_start \
                    and not FRONTBACK_RE.match(t):
                blocks.append((p, clean_title(t), False, None))

    blocks.sort(key=lambda b: b[0])
    # front matter before the first block
    if blocks and blocks[0][0] > 0:
        blocks.insert(0, (0, "Front matter", False, None))
    # label the preface block properly when we know it
    return blocks, sections


def build_from_intro_markers(entries, npages, heads):
    """Chapter map via chapter-intro splash pages (microbiology, psychology,
    statistics): outline gives titles/order, page text gives boundaries."""
    max_depth = max(d for d, _, _ in entries)
    cdepth = 0
    chap_titles = [t for d, t, p in entries
                   if d == cdepth and not FRONTBACK_RE.match(t)]
    sec_titles = [t for d, t, p in entries
                  if d == cdepth + 1 and t.lower() != "introduction"]
    # sections per chapter, in outline order
    chap_secs = []
    cur = -1
    for d, t, p in entries:
        if d == cdepth and not FRONTBACK_RE.match(t):
            cur += 1
            chap_secs.append([])
        elif d == cdepth + 1 and cur >= 0 and t.lower() != "introduction":
            chap_secs[cur].append(t)

    intro_pages = [i for i, h in enumerate(heads)
                   if h.startswith(INTRO_MARKER)]
    preface_p = next((p for d, t, p in entries
                      if d == cdepth and t.lower() == "preface" and p is not None),
                     None)

    def normkey(s):
        return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

    blocks, sections = [], []
    if preface_p is not None:
        blocks.append((preface_p, "Preface", False, None))
    cursor = 0
    chap_starts = []  # (start0, num, title) in order
    for ci, ctitle in enumerate(chap_titles):
        # first-section page for this chapter (forward text search)
        secpage = None
        if ci < len(chap_secs) and chap_secs[ci]:
            target = normkey(chap_secs[ci][0])[:45]
            for i in range(cursor, npages):
                if len(target) >= 6 and normkey(heads[i]).startswith(target):
                    secpage = i
                    break
        # chapter starts at the intro splash page shortly before its first
        # section, else right at the first section
        start = secpage
        if secpage is not None:
            for ip in reversed(intro_pages):
                if cursor <= ip <= secpage and secpage - ip <= 10:
                    start = ip
                    break
        if start is None:  # could not anchor: skip chapter (diagnostic)
            continue
        if start < cursor:
            start = cursor
        chap_starts.append((start, ci + 1, ctitle, ci))
        cursor = start + 1
    for bi, (start, num, ctitle, ci) in enumerate(chap_starts):
        end = chap_starts[bi + 1][0] if bi + 1 < len(chap_starts) else npages
        blocks.append((start, f"Chapter {num}: {ctitle}", True, num))
        # sections: forward scan for each section head inside the chapter
        sc = start
        for st in chap_secs[ci] if ci < len(chap_secs) else []:
            key = normkey(st)[:45]
            if len(key) < 6:
                continue
            for i in range(sc, end):
                if normkey(heads[i]).startswith(key):
                    sections.append((i, num, clean_title(st)))
                    sc = i + 1
                    break
    cursor = chap_starts[-1][0] + 1 if chap_starts else 0

    # back-matter blocks with known outline destinations (appendices etc.)
    for d, t, p in entries:
        if d == cdepth and FRONTBACK_RE.match(t) and p is not None \
                and t.lower() != "preface" and p >= cursor:
            am = APPENDIX_RE.match(t)
            label = f"Appendix {am.group(1)}: {am.group(2).strip()}" if am \
                else t
            blocks.append((p, label, False, None))

    blocks.sort(key=lambda b: b[0])
    if blocks and blocks[0][0] > 0:
        blocks.insert(0, (0, "Front matter", False, None))
    return blocks, sections


def build_physics(heads, npages):
    """College Physics: outline is unreadable; detect 'N TITLE' openers."""
    chapters = []
    for i, h in enumerate(heads):
        m = PHYS_CHAPTER_RE.match(h)
        if m:
            chapters.append((int(m.group(1)), smart_title(m.group(2).strip()), i))
    chapters.sort(key=lambda c: c[2])
    blocks = [(p0, f"Chapter {n}: {t}", True, n) for n, t, p0 in chapters]
    # back matter: appendix openers ("B SELECTED RADIOACTIVE ISOTOPES") + Index
    APPX_RE = re.compile(
        r"^([A-Z])\s+([A-Z][A-Z\s,&'\-–()]{2,40}?)"
        r"(?=\s+APPENDIX\s+[A-Z]\b|\s+[A-Z][a-z]|\s*\||\s*$)")
    if chapters:
        last_start = chapters[-1][2]
        for i in range(last_start, npages):
            h = heads[i]
            m = APPX_RE.match(h)
            if m and i > last_start:
                blocks.append((i, f"Appendix {m.group(1)}: "
                                  f"{smart_title(m.group(2).strip())}",
                               False, None))
            elif re.match(r"^Index\b", h) and i > last_start:
                blocks.append((i, "Index", False, None))
    blocks.sort(key=lambda b: b[0])
    if blocks and blocks[0][0] > 0:
        blocks.insert(0, (0, "Front matter", False, None))
    return blocks, []


def build_chapter_map(reader, pdf_file):
    heads = page_heads(reader)
    npages = len(reader.pages)
    if pdf_file == "college-physics.pdf":
        blocks, sections = build_physics(heads, npages)
        mode = "physics-text-scan"
    else:
        entries = outline_entries(reader)
        # Decide: outline destinations usable, or intro-marker fallback?
        max_depth = max((d for d, _, _ in entries), default=-1)
        cdepth = 1 if max_depth >= 2 else 0
        chap_dests = [p for d, t, p in entries
                      if d == cdepth and not FRONTBACK_RE.match(t)]
        usable = sum(1 for p in chap_dests if p)
        if entries and usable >= max(1, len(chap_dests) // 2):
            blocks, sections = build_from_outline(entries, npages, heads)
            mode = "outline"
        elif entries:
            blocks, sections = build_from_intro_markers(entries, npages, heads)
            mode = "intro-marker"
        else:
            blocks, sections = build_physics(heads, npages)
            mode = "text-scan-fallback"
    # per-page labels (0-based)
    page_label = ["Front matter"] * npages
    page_section = [None] * npages
    ordered = sorted(blocks, key=lambda b: b[0])
    for bi, (s0, label, is_ch, num) in enumerate(ordered):
        e0 = ordered[bi + 1][0] if bi + 1 < len(ordered) else npages
        for i in range(s0, min(e0, npages)):
            page_label[i] = label
    for p0, num, st in sections:
        if 0 <= p0 < npages and page_label[p0].startswith("Chapter "):
            # only attach the section inside its own chapter block
            if f"Chapter {num}:" in page_label[p0]:
                page_section[p0] = (num, st)
    # propagate section forward within the chapter until the next section
    last = None
    for i in range(npages):
        if page_section[i] is not None:
            last = page_section[i]
        elif not page_label[i].startswith("Chapter "):
            last = None
        elif last is not None and page_label[i].startswith(f"Chapter {last[0]}:"):
            page_section[i] = last
        else:
            last = None if not page_label[i].startswith("Chapter ") else last
    full = []
    for i in range(npages):
        lab = page_label[i]
        if page_section[i] is not None and lab.startswith("Chapter "):
            lab = f"{lab} — {page_section[i][1]}"
        full.append(lab)
    n_chap = sum(1 for _, _, is_ch, _ in ordered if is_ch)
    return full, mode, n_chap


def chunk_book(pdf_file: str, book_name: str, course: str):
    reader = PdfReader(str(DATA / pdf_file))
    page_label, mode, n_chap = build_chapter_map(reader, pdf_file)
    print(f"  mode={mode}, chapters={n_chap}")
    pages = extract_pages(DATA / pdf_file)
    print(f"  {book_name}: {len(pages)} content pages")
    chunks = []
    buf_words, buf_pages = [], []
    for pageno, text in pages:  # pageno is 1-based
        label = page_label[pageno - 1]
        # last-resort fallback: old "Chapter N" page-head regex
        if label in ("Front matter", "Unknown"):
            label = detect_chapter(text, label)
        words = text.split()
        buf_words.extend(words)
        buf_pages.append((pageno, label))
        while len(buf_words) >= CHUNK_WORDS:
            chunk_words = buf_words[:CHUNK_WORDS]
            first_pg, first_lab = buf_pages[0]
            last_pg, _ = buf_pages[-1]
            # majority vote on the chapter label across the chunk's pages
            votes = {}
            for _, lab in buf_pages:
                votes[lab] = votes.get(lab, 0) + 1
            best = max(votes, key=votes.get)
            chunks.append({
                "book": book_name, "course": course, "chapter": best,
                "page_start": first_pg, "page_end": last_pg,
                "text": " ".join(chunk_words),
            })
            buf_words = buf_words[CHUNK_WORDS - OVERLAP_WORDS:]
            buf_pages = buf_pages[-2:] if len(buf_pages) > 2 else buf_pages
    if len(buf_words) > 120:
        votes = {}
        for _, lab in buf_pages:
            votes[lab] = votes.get(lab, 0) + 1
        best = max(votes, key=votes.get) if votes else "Front matter"
        chunks.append({
            "book": book_name, "course": course, "chapter": best,
            "page_start": buf_pages[0][0], "page_end": buf_pages[-1][0],
            "text": " ".join(buf_words),
        })
    return chunks


def detect_chapter(text: str, current: str) -> str:
    """Last-resort fallback: old page-head 'Chapter N' regex."""
    head = " ".join(text.split()[:12])
    m = CHAPTER_RE.match(head)
    if m:
        title = m.group(2).strip()[:60]
        return f"Chapter {m.group(1)}" + (f": {title}" if title else "")
    return current


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

    # ---- per-book chapter-label stats ----
    print("\n== chapter-label coverage ==")
    for course, g in df.groupby("course"):
        real = g["chapter"].str.startswith("Chapter ").mean() * 100
        n = len(g)
        samples = g.loc[g["chapter"].str.startswith("Chapter "),
                        "chapter"].drop_duplicates().head(4).tolist()
        print(f"{course}: {n} chunks, {real:.1f}% with real chapter label")
        for s in samples:
            print(f"    e.g. {s[:100]}")
        front = (g["chapter"] == "Front matter").mean() * 100
        other = sorted(g.loc[~g["chapter"].str.startswith("Chapter "),
                             "chapter"].unique().tolist())
        print(f"    Front matter: {front:.1f}% | other labels: {other}")

    print("\nEmbedding chunks (all-MiniLM-L6-v2) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(df["text"].tolist(), batch_size=64,
                       show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)
    assert emb.shape[1] == 384, f"unexpected embedding dim {emb.shape[1]}"
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(ART / "kb.index"))
    print(f"Saved: {ART/'chunks.parquet'} ({len(df)} chunks), {ART/'kb.index'}")
    print(df.groupby("course").size().to_string())


if __name__ == "__main__":
    main()
