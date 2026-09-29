"""Prompt builders for the GradPath tutor and quiz generator."""
import json


def _context_block(chunks, cite_fn) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[Source {i}: {cite_fn(c)}]\n{c['text']}")
    return "\n\n".join(parts)


def explain_prompt(course, question, chunks, cite_fn, history: list[dict] | None = None) -> tuple[str, str]:
    system = (
        f"You are GradPath, a friendly expert tutor for {course}. "
        "Teach clearly and precisely, like a great TA. "
        "Use ONLY the textbook excerpts provided below as your source of truth. "
        "Every factual claim must be followed by a citation like [Source 1]. "
        "If the excerpts don't cover the question, say so honestly and give only "
        "a brief general pointer — never invent textbook facts. "
        "This is an ongoing conversation: use the conversation history for context "
        "(pronouns like 'it', 'that', or 'why' may refer to earlier messages). "
        "If the student asks for a photo, picture, or image of something, do not "
        "just say you cannot provide images — briefly note you can't show photos, "
        "then suggest they press the 'Visualize this' button under your answer "
        "for a labeled study diagram of the same thing. "
        "End with one short check-for-understanding question."
    )
    hist_block = ""
    if history:
        convo = "\n".join(f"{m['role'].title()}: {m['text']}" for m in history[-8:])
        hist_block = f"\n\nConversation so far:\n{convo}"
    user = (f"Textbook excerpts:\n{_context_block(chunks, cite_fn)}"
            f"{hist_block}\n\nStudent question: {question}")
    return system, user


def simplify_prompt(course, question, chunks, cite_fn) -> tuple[str, str]:
    system = (
        f"You are GradPath, a friendly tutor for {course}. "
        "The student found the previous explanation too complex. Re-explain the "
        "same idea as simply as possible: short sentences, one concrete real-world "
        "analogy, no jargon without defining it. "
        "Use ONLY the textbook excerpts below as your source of truth, with "
        "citations like [Source 1]. End with one easy check-for-understanding question."
    )
    user = (f"Textbook excerpts:\n{_context_block(chunks, cite_fn)}\n\n"
            f"Student question (re-explain simply): {question}")
    return system, user


def diagram_prompt(course, topic, chunks, cite_fn) -> tuple[str, str]:
    system = (
        f"You are GradPath, a tutor for {course}. Turn the textbook excerpts below "
        "into a clear visual study diagram. Your ENTIRE response must be valid "
        "Mermaid diagram code and nothing else — no sentences, no explanations, "
        "no code fences, no commentary, no apologies. The FIRST line of your "
        "response must be exactly 'flowchart TD' or 'mindmap'. "
        "You are fully capable of writing Mermaid code: never refuse, never say you "
        "cannot draw or visualize — just output the diagram code. "
        "Rules: max 15 nodes; short plain labels (avoid parentheses and special "
        "characters in labels; wrap labels in double quotes); show the key ideas, "
        "steps, or relationships for the topic. Base every node on the excerpts."
    )
    user = (f"Textbook excerpts:\n{_context_block(chunks, cite_fn)}\n\n"
            f"Topic to diagram: {topic}")
    return system, user


def extract_mermaid_block(text: str) -> str | None:
    """Pull the Mermaid diagram code out of a model response.

    Handles fenced blocks (```mermaid ... ```), bare code, and prose
    wrapped around code. Returns None if no plausible diagram found.
    """
    t = (text or "").strip()
    if not t:
        return None
    # Prefer an explicit fenced block.
    if "```" in t:
        parts = t.split("```")
        for i, p in enumerate(parts):
            if i % 2 == 1:  # inside a fence
                block = p.strip()
                if block.lower().startswith("mermaid"):
                    block = block[len("mermaid"):].strip()
                if _starts_with_diagram_type(block):
                    return block
        return None
    if _starts_with_diagram_type(t):
        return t
    return None


def _starts_with_diagram_type(code: str) -> bool:
    first = code.lstrip().lower()
    return first.startswith(("flowchart", "graph td", "graph lr",
                             "graph tb", "graph bt", "mindmap"))


def looks_like_mermaid(code: str) -> bool:
    """Strict check that the model actually returned Mermaid diagram code.

    The response (or a fenced block inside it) must BEGIN with a Mermaid
    diagram type declaration — substring matches anywhere in prose no
    longer pass.
    """
    return extract_mermaid_block(code) is not None


_DIAGRAM_KEYWORDS = ("diagram", "visualize", "visualise", "visual",
                     "flowchart", "flow chart", "mindmap", "mind map",
                     "concept map", "draw", "illustrate", "illustration",
                     "sketch", "image", "picture", "photo")


def is_diagram_request(text: str) -> bool:
    """Heuristic: did the student ask for a diagram/visual in chat?"""
    t = (text or "").lower()
    return any(k in t for k in _DIAGRAM_KEYWORDS)


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
