"""GradPath — an AI study copilot for university students.

Pillars:
  1. Tutor: RAG over real OpenStax textbooks (Explain or Socratic mode), answers with citations.
  2. Quiz Me: exam-style questions generated from the textbook, auto-graded.
  3. GPA Planner: what grades you need to hit your target GPA.

The tutor/quiz need a free Gemini API key (https://aistudio.google.com/),
entered in the sidebar. The GPA planner works without any key.
"""
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import html as htmlmod

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rag import TextbookRetriever          # noqa: E402
from llm import make_client                # noqa: E402
import tutor as tutor_prompts              # noqa: E402
import gpa as gpa_math                     # noqa: E402

st.set_page_config(page_title="GradPath — AI Study Copilot", page_icon="🎓", layout="wide")


def render_mermaid(code: str, height: int = 520):
    """Render Mermaid diagram code (concept map / flowchart) in the app."""
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        lines = lines[1:] if len(lines) > 1 else []
        code = "\n".join(lines).rsplit("```", 1)[0]
    if code.lstrip().startswith("mermaid"):
        code = code.lstrip()[len("mermaid"):].strip()
    safe = htmlmod.escape(code)
    page = f"""<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<pre class="mermaid" style="background:#fff;border-radius:12px;padding:16px;">{safe}</pre>
<script>mermaid.initialize({{startOnLoad:true, theme:'default'}});</script>"""
    components.html(page, height=height, scrolling=True)

# ---------------------------------------------------------------- styles
st.markdown("""
<style>
.hero { background: linear-gradient(135deg,#1a2a6c,#b21f1f,#fdbb2d); border-radius:16px;
        padding:28px 32px; color:white; margin-bottom:18px; }
.hero h1 { margin:0; font-size:2.1rem; }
.hero p { margin:6px 0 0 0; opacity:.92; }
.cite { font-size:.8rem; color:#555; background:#f1f3f5; border-radius:8px; padding:8px 12px; margin-top:8px;}
.score-good { color:#2b8a3e; font-weight:700; } .score-bad { color:#c92a2a; font-weight:700; }
</style>
<div class="hero"><h1>🎓 GradPath</h1>
<p>Your AI study copilot — a tutor grounded in your actual textbooks, auto-generated quizzes,
and a GPA planner that keeps your goals on track.</p></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading textbook knowledge base…")
def get_retriever():
    return TextbookRetriever()

try:
    retr = get_retriever()
except Exception as e:
    st.error("Textbook knowledge base not found. Run `python src/build_index.py` first.")
    st.stop()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Setup")
    provider = st.selectbox("AI provider", ["Gemini", "OpenAI"],
                            help="Gemini keys are free at aistudio.google.com. "
                                 "OpenAI keys are at platform.openai.com/api-keys.")
    api_key = st.text_input(f"{provider} API key", type="password",
                            help="Needed for Tutor & Quiz. Never stored — only used for your requests.")
    if not api_key:
        key_link = ("[aistudio.google.com](https://aistudio.google.com/)" if provider == "Gemini"
                    else "[platform.openai.com/api-keys](https://platform.openai.com/api-keys)")
        st.caption(f"👉 Get a key at {key_link} and paste it here. "
                   "The GPA planner works without one.")
    st.divider()
    course = st.selectbox("📖 Course", retr.courses)
    st.caption("Built for UH courses — answers are grounded in the OpenStax textbook for this class, with citations.")

tab_tutor, tab_quiz, tab_gpa = st.tabs(["💬 Tutor", "📝 Quiz Me", "🎯 GPA Planner"])

# ================================================================ TUTOR
with tab_tutor:
    mode = st.radio("Tutoring mode", ["Explain", "Socratic"], horizontal=True,
                    help="Explain: teaches directly. Socratic: guides you with questions instead of answers.")
    if "chat" not in st.session_state:
        st.session_state.chat = []

    # --- follow-up actions from the buttons under the last answer ---
    fu = st.session_state.pop("followup", None)
    if fu:
        last = st.session_state.get("last_qa", {})
        if not api_key:
            st.warning("Add your free Gemini API key in the sidebar first.")
        elif not last:
            st.warning("Ask a question first, then use the follow-up buttons.")
        else:
            try:
                client = make_client(provider, api_key)
                if fu == "simpler":
                    system, user = tutor_prompts.simplify_prompt(
                        last["course"], last["q"], last["chunks"], TextbookRetriever.cite)
                    with st.spinner("Simplifying…"):
                        ans = client.generate(system, user, temperature=0.5)
                    st.session_state.chat.append(
                        {"role": "assistant", "text": ans, "cites": last["cites"]})
                elif fu == "visualize":
                    system, user = tutor_prompts.diagram_prompt(
                        last["course"], last["q"], last["chunks"], TextbookRetriever.cite)
                    with st.spinner("Drawing your diagram…"):
                        code = client.generate(system, user, temperature=0.3, max_tokens=1200)
                        if not tutor_prompts.looks_like_mermaid(code):
                            # Model dodged the diagram request — one firmer retry.
                            retry_system = (system + " Your previous reply was not valid "
                                            "Mermaid code. Reply NOW with ONLY the Mermaid "
                                            "diagram code and nothing else.")
                            code = client.generate(retry_system, user,
                                                   temperature=0.2, max_tokens=1200)
                    if tutor_prompts.looks_like_mermaid(code):
                        st.session_state.diagram = {"topic": last["q"], "code": code}
                    else:
                        st.session_state.diagram = None
                        st.error("Couldn't generate a diagram for this one — "
                                 "try pressing 🖼️ Visualize this again.")
                elif fu == "quizme":
                    st.session_state.quiz_topic = last["q"]
                    st.info("📝 Topic sent to the Quiz Me tab — open it and hit **Generate quiz**.")
            except Exception as e:
                st.error(f"Tutor error: {e}")

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["text"])
            if m.get("cites"):
                st.markdown(f"<div class='cite'>📚 Sources:<br>{'<br>'.join(m['cites'])}</div>",
                            unsafe_allow_html=True)

    # --- one-tap follow-ups under the latest answer ---
    if (st.session_state.chat and st.session_state.chat[-1]["role"] == "assistant"
            and st.session_state.get("last_qa")):
        b1, b2, b3 = st.columns(3)
        if b1.button("🔍 Explain simpler", key="fu_simpler"):
            st.session_state.followup = "simpler"
            st.rerun()
        if b2.button("🖼️ Visualize this", key="fu_visual"):
            st.session_state.followup = "visualize"
            st.rerun()
        if b3.button("📝 Quiz me on this", key="fu_quiz"):
            st.session_state.followup = "quizme"
            st.rerun()

    if st.session_state.get("diagram"):
        d = st.session_state.diagram
        with st.expander(f"🖼️ Visual: {d['topic'][:80]}", expanded=True):
            try:
                render_mermaid(d["code"])
            except Exception as e:
                st.error(f"Couldn't render the diagram: {e}")
            if st.button("Close visual"):
                st.session_state.diagram = None
                st.rerun()

    q = st.chat_input(f"Ask about {course}…")
    if q:
        if not api_key:
            st.warning("Add your free Gemini API key in the sidebar to chat with the tutor.")
            st.stop()
        st.session_state.chat.append({"role": "user", "text": q})
        with st.chat_message("user"):
            st.markdown(q)
        chunks = retr.search(q, course, k=5)
        cites = [f"[{i+1}] {TextbookRetriever.cite(c)}" for i, c in enumerate(chunks)]
        try:
            client = make_client(provider, api_key)
            if mode == "Explain":
                system, user = tutor_prompts.explain_prompt(
                    course, q, chunks, TextbookRetriever.cite,
                    [{"role": m["role"], "text": m["text"]} for m in st.session_state.chat[:-1]])
                temp = 0.4
            else:
                system, user = tutor_prompts.socratic_prompt(
                    course, q, chunks, TextbookRetriever.cite,
                    [{"role": m["role"], "text": m["text"]} for m in st.session_state.chat[:-1]])
                temp = 0.6
            with st.spinner("Thinking…"):
                ans = client.generate(system, user, temperature=temp)
        except Exception as e:
            st.error(f"Tutor error: {e}")
            st.stop()
        st.session_state.chat.append({"role": "assistant", "text": ans, "cites": cites})
        st.session_state.last_qa = {"q": q, "chunks": chunks, "cites": cites, "course": course}
        with st.chat_message("assistant"):
            st.markdown(ans)
            st.markdown(f"<div class='cite'>📚 Sources:<br>{'<br>'.join(cites)}</div>",
                        unsafe_allow_html=True)
    if st.session_state.chat and st.button("Clear conversation"):
        st.session_state.chat = []
        st.session_state.last_qa = None
        st.session_state.diagram = None
        st.rerun()

# ================================================================ QUIZ
with tab_quiz:
    st.subheader(f"Test yourself on {course}")
    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input("Topic", value=st.session_state.get("quiz_topic", ""),
                              placeholder="e.g. cellular respiration, hypothesis testing, derivatives")
    with col2:
        n_q = st.slider("# questions", 3, 8, 5)
    if st.button("Generate quiz", type="primary"):
        if not api_key:
            st.warning("Add your free Gemini API key in the sidebar to generate quizzes.")
            st.stop()
        if not topic.strip():
            st.warning("Enter a topic first.")
            st.stop()
        chunks = retr.search(topic, course, k=6)
        try:
            client = make_client(provider, api_key)
            system, user = tutor_prompts.quiz_prompt(course, topic, chunks, TextbookRetriever.cite, n_q)
            with st.spinner("Writing questions from your textbook…"):
                raw = client.generate(system, user, temperature=0.5, max_tokens=2500)
            st.session_state.quiz = tutor_prompts.parse_quiz(raw)
            st.session_state.quiz_topic = topic
        except Exception as e:
            st.error(f"Quiz error: {e} — try a more specific topic.")
    quiz = st.session_state.get("quiz")
    if quiz:
        st.markdown(f"**Topic:** {st.session_state.get('quiz_topic','')}")
        answers = []
        for i, item in enumerate(quiz):
            st.markdown(f"**Q{i+1}. {item['question']}**")
            sel = st.radio(f"q{i}", item["choices"], key=f"quiz_{i}", label_visibility="collapsed")
            answers.append(item["choices"].index(sel))
        if st.button("Grade quiz"):
            score = sum(a == item["answer"] for a, item in zip(answers, quiz))
            pct = 100 * score / len(quiz)
            cls = "score-good" if pct >= 70 else "score-bad"
            st.markdown(f"<span class='{cls}'>Score: {score}/{len(quiz)} ({pct:.0f}%)</span>",
                        unsafe_allow_html=True)
            for i, item in enumerate(quiz):
                ok = answers[i] == item["answer"]
                icon = "✅" if ok else "❌"
                with st.expander(f"{icon} Q{i+1}. {item['question'][:70]}…"):
                    st.markdown(f"**Correct:** {item['choices'][item['answer']]}")
                    if not ok:
                        st.markdown(f"**You chose:** {item['choices'][answers[i]]}")
                    st.caption(item["explanation"])

# ================================================================ GPA
with tab_gpa:
    st.subheader("What will it take to hit your target?")
    c1, c2, c3 = st.columns(3)
    with c1:
        cur_gpa = st.number_input("Current GPA", 0.0, 4.0, 3.20, 0.05)
    with c2:
        done_credits = st.number_input("Credits completed", 0.0, 200.0, 45.0, 1.0)
    with c3:
        target = st.number_input("Target GPA", 0.0, 4.0, 3.70, 0.05,
                                 help="e.g. 3.7 for med school, 3.5 for competitive internships")
    st.markdown("**Planned courses**")
    import pandas as pd
    if "courses_df" not in st.session_state:
        st.session_state.courses_df = pd.DataFrame([
            {"Course": "Microbiology", "Credits": 4},
            {"Course": "Genetics", "Credits": 3},
            {"Course": "Calculus II", "Credits": 4},
            {"Course": "Psychology", "Credits": 3},
        ])
    cdf = st.data_editor(st.session_state.courses_df, num_rows="dynamic", use_container_width=True)
    planned_credits = float(cdf["Credits"].sum()) if len(cdf) else 0.0

    if planned_credits > 0:
        req = gpa_math.required_term_gpa(cur_gpa, done_credits, target, planned_credits)
        st.metric(f"GPA needed across your next {planned_credits:.0f} credits", f"{req:.2f}")
        st.info(gpa_math.feasibility(req))
        st.markdown("**What-if: pick expected grades**")
        planned = []
        cols = st.columns(min(len(cdf), 4))
        for i, row in cdf.iterrows():
            with cols[i % len(cols)]:
                g = st.selectbox(f"{row['Course']} ({row['Credits']} cr)",
                                 list(gpa_math.GRADE_POINTS), index=0, key=f"g_{i}")
                planned.append({"course": row["Course"], "credits": float(row["Credits"]), "grade": g})
        proj = gpa_math.projected_gpa(cur_gpa, done_credits, planned)
        delta = proj - target
        st.metric("Projected cumulative GPA", f"{proj:.3f}",
                  delta=f"{delta:+.3f} vs target",
                  delta_color="normal" if delta >= 0 else "inverse")
        with st.expander("Minimum grade per course (if everything else is an A)"):
            mins = gpa_math.min_grades_for_target(cur_gpa, done_credits, target, planned)
            st.table(pd.DataFrame(
                [{"Course": m["course"], "Credits": m["credits"], "Minimum grade": m["min_grade"]}
                 for m in mins]))

st.caption("Textbooks: OpenStax (openstax.org), used under CC BY. Tutor answers cite the textbook — "
           "always verify critical facts. GradPath is a study aid, not academic advice.")
