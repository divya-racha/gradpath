# 🎓 GradPath — AI Study Copilot for University Students

**Live demo:** *(link after deployment)*

GradPath helps students actually learn their coursework *and* stay on track for their goals —
whether that's a GPA for internships, grad school, or med school.

## What it does

1. **💬 Tutor (RAG, not a generic chatbot)** — Ask questions about Microbiology, Psychology,
   Biology/Genetics, Calculus I, or Statistics. Answers are retrieved from the real
   [OpenStax](https://openstax.org) textbook for that course and come with **citations**
   (book, chapter, pages). Two modes:
   - *Explain* — teaches the concept directly, ends with a check-for-understanding question
   - *Socratic* — never gives the answer; guides you with one focused question at a time
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

- **Knowledge base:** 5 OpenStax textbooks (CC BY licensed), chunked with page/chapter metadata
- **Retrieval:** `sentence-transformers/all-MiniLM-L6-v2` + FAISS inner-product search, filtered by course
- **Generation:** Gemini 2.0 Flash (free tier) with strict ground-in-the-excerpts system prompts
- **GPA math:** closed-form — no LLM involved

## Run it locally

```bash
pip install -r requirements.txt
python src/build_index.py        # one-time: builds artifacts/kb.index (~10 min)
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
│   ├── build_index.py    # PDF → chunks → embeddings → FAISS
│   ├── rag.py            # course-filtered textbook retriever
│   ├── llm.py            # minimal Gemini REST client
│   ├── tutor.py          # prompts: explain / socratic / quiz generation
│   └── gpa.py            # GPA planner math (pure functions)
├── artifacts/            # kb.index + chunks.parquet (built, not in git)
└── data/                 # OpenStax PDFs (downloaded, not in git)
```

## Notes & honesty

- Textbook content is © OpenStax, used under CC BY — always verify critical facts.
- Quiz/test-prep questions are *generated from textbook content*, not copied from real
  LSAT/MCAT banks (those are copyrighted).
- GradPath is a study aid, not academic advice. It won't write your essays or take your exams.

Built by Sree Divya — a UH CS junior building the study tool she wished existed.
