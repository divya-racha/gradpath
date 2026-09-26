"""Prompt builders for the GradPath tutor and quiz generator."""
import json


def _context_block(chunks, cite_fn) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[Source {i}: {cite_fn(c)}]\n{c['text']}")
    return "\n\n".join(parts)


def explain_prompt(course, question, chunks, cite_fn) -> tuple[str, str]:
    system = (
        f"You are GradPath, a friendly expert tutor for {course}. "
        "Teach clearly and precisely, like a great TA. "
        "Use ONLY the textbook excerpts provided below as your source of truth. "
        "Every factual claim must be followed by a citation like [Source 1]. "
        "If the excerpts don't cover the question, say so honestly and give only "
        "a brief general pointer — never invent textbook facts. "
        "End with one short check-for-understanding question."
    )
    user = f"Textbook excerpts:\n{_context_block(chunks, cite_fn)}\n\nStudent question: {question}"
    return system, user


def socratic_prompt(course, question, chunks, cite_fn, history: list[dict]) -> tuple[str, str]:
    system = (
        f"You are GradPath, a Socratic tutor for {course}. "
        "NEVER give the direct answer. Instead, guide the student with ONE "
        "focused guiding question at a time, building on the textbook excerpts "
        "below. Acknowledge good reasoning warmly; gently redirect mistakes by "
        "pointing at the relevant excerpt. Keep replies short (2-4 sentences). "
        "If the student is stuck after 3 exchanges, offer a small hint from the excerpts."
    )
    convo = "\n".join(f"{m['role'].title()}: {m['text']}" for m in history[-6:])
    user = (f"Textbook excerpts:\n{_context_block(chunks, cite_fn)}\n\n"
            f"Conversation so far:\n{convo}\n\nStudent's latest message: {question}\n"
            "Respond with your next single guiding question (or brief feedback + question).")
    return system, user


def quiz_prompt(course, topic, chunks, cite_fn, n: int = 5) -> tuple[str, str]:
    system = (
        f"You are GradPath, a quiz author for {course}. "
        "Write exam-style multiple-choice questions using ONLY the textbook "
        "excerpts below. Each question must be answerable from the excerpts. "
        "Respond with ONLY a JSON array, no other text. Each item:\n"
        '{"question": "...", "choices": ["A) ...", "B) ...", "C) ...", "D) ..."], '
        '"answer": 0-3, "explanation": "one sentence citing the excerpt"}'
    )
    user = (f"Textbook excerpts:\n{_context_block(chunks, cite_fn)}\n\n"
            f"Write {n} questions about: {topic}")
    return system, user


def parse_quiz(raw: str) -> list[dict]:
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```")[1]
        if s.lstrip().startswith("json"):
            s = s.lstrip()[4:]
    items = json.loads(s)
    out = []
    for it in items:
        out.append({
            "question": it["question"],
            "choices": it["choices"][:4],
            "answer": int(it["answer"]),
            "explanation": it.get("explanation", ""),
        })
    return out
