# 🎓 GradPath — AI Study Copilot for University Students

**Live demo:** https://gradpath-727nuzefxhbofh3rmobmk3.streamlit.app/

GradPath helps students actually learn their coursework *and* stay on track for their goals —
whether that's a GPA for internships, grad school, or med school.

## What it does

1. **💬 Tutor (RAG, not a generic chatbot)** — Ask questions about your course. Answers are
   retrieved from the real [OpenStax](https://openstax.org) textbook for that class and come
   with **citations** (book, chapter, pages). Two modes:
   - *Explain* — teaches the concept directly, ends with a check-for-understanding question
   - *Socratic* — never gives the answer; guides you with one focused question at a time

   Covered courses (UH labels): **BIOL 1306** Biology, **BIOL 2321** Microbiology,
   **CHEM 1311/1312** Chemistry, **MATH 1342** Statistics, **MATH 2413** Calculus I,
   **PHYS 1301** Physics, **PSYC 2301** Psychology.
2. **📝 Quiz Me** — Generates exam-style multiple-choice questions from the textbook chapter
   covering your topic, then auto-grades with explanations.
3. **🎯 GPA Planner** — Enter your current GPA, completed credits, and target (e.g. 3.7 for
   med school). GradPath computes the GPA you need across your planned courses, shows
   what-if scenarios per course, and tells you honestly when a target isn't reachable.

## Why not just use ChatGPT?

Generic chatbots don't know *your* textbook, can't cite it, and happily invent facts.
GradPath grounds every answer in the actual course material via retrieval-augmented
generation, cites its sources, and refuses to invent textbook facts it can't find.

## How it works

```
OpenStax PDFs → text extraction → 450-word chunks → MiniLM embeddings → FAISS index
                                                                              ↓
Student question → retrieve top-5 chunks (course-filtered) → Gemini 2.0 Flash
                   → grounded answer + citations
```

- **Knowledge base:** 7 OpenStax textbooks (CC BY licensed) → **7,856 chunks** with
  chapter/section labels extracted from each PDF's table of contents
  (86.7–99.4% of chunks carry a real chapter label per book)
- **Retrieval:** `sentence-transformers/all-MiniLM-L6-v2` + FAISS inner-product search, filtered by course
- **Generation:** provider dropdown — Gemini 2.0 Flash (free tier, default) or OpenAI
  `gpt-4o-mini`, with strict ground-in-the-excerpts system prompts
- **GPA math:** closed-form — no LLM involved

## Measured retrieval quality

An offline eval harness (`src/eval/`) scores the retriever on **105 original questions**
(15 per subject) — no LLM calls, fully reproducible via `python src/eval/run_eval.py`:

| Subject | P@5 | MRR | Keyword coverage@5 |
|---|---|---|---|
| BIOL 1306 · Biology | 0.733 | 0.833 | 0.844 |
| BIOL 2321 · Microbiology | 0.613 | 0.680 | 0.867 |
| CHEM 1311/1312 · Chemistry | 0.893 | 0.967 | 0.889 |
| MATH 1342 · Statistics | 0.840 | 1.000 | 0.867 |
| MATH 2413 · Calculus I | 0.853 | 0.889 | 0.911 |
| PHYS 1301 · Physics | 0.920 | 0.900 | 0.844 |
| PSYC 2301 · Psychology | 0.747 | 0.867 | 0.867 |
| **Overall** | **0.800** | **0.877** | **0.870** |

P@5 = fraction of top-5 chunks from the right chapter; MRR = rank of the first good hit;
keyword coverage = fraction of distinctive expected-chapter phrases found in the top-5 text.
Full report: `artifacts/eval_report.md`. A CI workflow re-runs the eval on every push.

## Run it locally

```bash
pip install -r requirements.txt
python src/build_index.py        # one-time: builds artifacts/kb.index
streamlit run app/app.py
```

Get a **free** Gemini API key at https://aistudio.google.com/ (or use your own OpenAI key
from https://platform.openai.com/api-keys) and paste it in the sidebar —
pick the matching provider. The GPA planner needs no key.

## Project structure

```
gradpath/
├── app/app.py            # Streamlit UI (tutor chat, quiz, GPA planner)
├── src/
│   ├── build_index.py    # PDF → chunks → embeddings → FAISS (outline-based chapters)
│   ├── rag.py            # course-filtered textbook retriever
│   ├── llm.py            # minimal Gemini/OpenAI REST client
│   ├── tutor.py          # prompts: explain / socratic / quiz generation / diagrams
│   ├── gpa.py            # GPA planner math (pure functions)
│   ├── test_retrieval.py # retrieval smoke test
│   └── eval/             # offline eval harness (105-question set + runner)
├── .github/workflows/    # CI: compile check + smoke test + eval on push
├── artifacts/            # kb.index + chunks.parquet + eval_report.md (in git)
└── data/                 # OpenStax PDFs (downloaded locally, not in git)
```

## Notes & honesty

- Textbook content is © OpenStax, used under CC BY — always verify critical facts.
- Quiz/test-prep questions are *generated from textbook content*, not copied from real
  LSAT/MCAT banks (those are copyrighted).
- GradPath is a study aid, not academic advice. It won't write your essays or take your exams.

Built by Sree Divya — a UH CS junior building the study tool she wished existed.
