"""
Personal study assistant: a Gemini agent whose tools are PaperMind's own NLP + GenAI functions.

It answers from data (weightage, hot topics, past questions), sets practice questions, checks answers against
the generated key, and records every graded attempt so advice is personalised: priority = how much a topic
weighs in the exam x how weak this student is at it.
"""
import json
from collections import defaultdict

from papermind import config, db, generate, llm, nlp
from papermind.analysis import analyze

SYSTEM = """You are PaperMind, a personal exam-preparation coach for Indian engineering students
(GATE CS, GATE DA, and university mid-sem/end-sem exams). Areas: {areas}.

Rules:
- Ground every claim about exam patterns in tool results (weightage, hot topics, past questions). Cite years.
- When the student wants practice, create a practice set and ask the questions ONE at a time. Never reveal an
  answer before the student attempts it; then fetch the official key with get_answer, explain, and call
  record_result. Use the student's progress to choose what to practise next.
- For study plans, combine syllabus weightage with my_progress (weak + high-weight topics first).
- Write maths in LaTeX ($...$). Be concise and encouraging; prefer bullet points."""


def make_tools(con, student: str) -> list:
    """Tools are closures over the DB connection and student; type hints + docstrings define the schema."""

    def syllabus_overview(area: str) -> dict:
        """Marks weightage of each unit/subject in past papers of an area (e.g. gate_cs, gate_da), with trends."""
        a = analyze(db.questions(con, area))
        return {"papers": a.get("papers"), "years": a.get("years"), "question_types_per_paper": a.get("question_types"),
                "units": [{k: u[k] for k in ("unit", "avg_marks_per_paper", "share_pct", "trend")} for u in a.get("units", [])]}

    def hot_topics(area: str, unit: str = "", top: int = 10) -> list[dict]:
        """Most important topics by recency-weighted marks. Optionally restrict to one unit name."""
        rows = [r for r in db.questions(con, area) if not unit or r["unit"] == unit]
        return analyze(rows).get("topics", [])[:top]

    def search_past_questions(area: str, query: str, unit: str = "", k: int = 5) -> list[dict]:
        """Find real past exam questions similar to a query (concept, keyword or question text)."""
        rows = [r for r in db.questions(con, area) if not unit or r["unit"] == unit]
        if not rows:
            return []
        sims = nlp.similarity([query], [f"{r['topic']} {r['text']}" for r in rows])[0]
        best = sims.argsort()[::-1][:k]
        return [{"paper": rows[i]["title"], "q": rows[i]["qno"], "type": rows[i]["qtype"], "marks": rows[i]["marks"],
                 "topic": rows[i]["topic"], "text": rows[i]["text"][:600], "options": rows[i]["options"],
                 "answer": rows[i]["answer"]} for i in best]

    def create_practice_set(request: str) -> dict:
        """Generate a new practice set or paper from a natural-language request, e.g.
        '5 questions on deadlock for GATE CS' or 'mid-sem paper for DBMS'. Returns questions WITHOUT answers."""
        req = generate.parse_request(request, nlp.areas())
        paper = generate.generate_paper(req, db.questions(con, req.area))
        pid = con.execute("INSERT INTO generated (request, paper) VALUES (?, ?)", (request, json.dumps(paper))).lastrowid
        con.commit()
        return {"paper_id": pid, "title": paper["title"], "area": paper["area"],
                "questions": [{"n": q["n"], "type": q["qtype"], "marks": q["marks"], "unit": q["unit"], "topic": q["topic"],
                               "text": q["text"], "options": q["options"]} for s in paper["sections"] for q in s["questions"]]}

    def get_answer(paper_id: int, question_number: int) -> dict:
        """Official answer key and worked explanation for one question of a generated set (use after the attempt)."""
        row = con.execute("SELECT paper FROM generated WHERE id = ?", (paper_id,)).fetchone()
        if not row:
            return {"error": "no such paper"}
        for s in json.loads(row["paper"])["sections"]:
            for q in s["questions"]:
                if q["n"] == question_number:
                    return {k: q.get(k) for k in ("qtype", "answer", "explanation", "unit", "topic", "source", "ref")}
        return {"error": "no such question"}

    def record_result(area: str, unit: str, topic: str, correct: bool, source: str = "") -> str:
        """Record whether the student answered a question correctly (call after every graded attempt)."""
        con.execute("INSERT INTO attempts (student, area, unit, topic, correct, source) VALUES (?,?,?,?,?,?)",
                    (student, area, unit, topic, int(correct), source))
        con.commit()
        return "recorded"

    def my_progress(area: str = "") -> dict:
        """The student's accuracy per unit/topic and a priority list (weak x exam weight) of what to study next."""
        return progress(con, student, area)

    return [syllabus_overview, hot_topics, search_past_questions, create_practice_set, get_answer, record_result, my_progress]


def progress(con, student: str, area: str = "") -> dict:
    sql, args = "SELECT * FROM attempts WHERE student = ?", [student]
    if area:
        sql += " AND area = ?"
        args.append(area)
    rows = [dict(r) for r in con.execute(sql, args)]
    by_topic = defaultdict(lambda: [0, 0])
    for r in rows:
        t = by_topic[(r["area"], r["unit"], r["topic"])]
        t[0] += r["correct"]
        t[1] += 1
    hot = {}
    for a in {k[0] for k in by_topic}:
        hot |= {(a, t["unit"], t["topic"]): t["hot_score"] for t in analyze(db.questions(con, a)).get("topics", [])}
    topics = []
    for k, (c, n) in by_topic.items():
        acc = (c + 1) / (n + 2)  # Laplace-smoothed: one wrong answer is not 'zero accuracy'
        topics.append({"area": k[0], "unit": k[1], "topic": k[2], "attempts": n, "correct": c,
                       "accuracy": round(c / n, 2), "priority": round((1 - acc) * (hot.get(k, 10) + 10), 1)})
    topics.sort(key=lambda t: -t["priority"])
    return {"attempts": len(rows), "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 2) if rows else None,
            "topics": topics}


def chat(con, student: str, message: str) -> str:
    """One conversational turn; history (including tool calls) persists per student."""
    if not llm.available():
        return "The assistant needs GEMINI_API_KEY in .env."
    from google.genai import types

    row = con.execute("SELECT history FROM chats WHERE student = ?", (student,)).fetchone()
    history = [types.Content.model_validate(c) for c in json.loads(row["history"])] if row else []
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM.format(areas=", ".join(nlp.areas())),
        tools=make_tools(con, student),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=8),
        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
        temperature=0.6,
    )
    last = None
    for model in [m for m in config.GEN_MODELS if m not in llm._exhausted]:
        try:
            session = llm._client().chats.create(model=model, config=cfg, history=history[-40:])
            reply = session.send_message(message).text or "(no reply)"
            break
        except Exception as e:  # daily quota on this model: try the next one
            last = e
            if "per day" in str(e) or "PerDay" in str(e) or "429" in str(e):
                llm._exhausted.add(model)
                continue
            raise
    else:
        raise llm.LLMUnavailable(f"No model available: {last}")
    hist = [c.model_dump(mode="json", exclude_none=True) for c in session.get_history(curated=True)]
    con.execute("INSERT INTO chats (student, history, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(student) DO UPDATE SET history = excluded.history, updated_at = CURRENT_TIMESTAMP",
                (student, json.dumps(hist)))
    con.commit()
    return reply


def reset(con, student: str) -> None:
    con.execute("DELETE FROM chats WHERE student = ?", (student,))
    con.commit()
