"""
GenAI paper generator.

  student request ──parse──▶ ExamRequest ──blueprint──▶ slots (marks, type, section)
        ──plan (history-weighted topic sampling)──▶ slot topics
        ──retrieve (past questions on that topic = RAG exemplars)──▶ Gemini writes new questions
        ──validate (format, answer key, novelty vs. corpus, marks total)──▶ paper

Without a Gemini key the same plan is filled with real past questions ("PYQ mock").
"""
import math
import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import BaseModel, Field

from papermind import analysis, config, llm
from papermind.nlp import BLOOM_LEVELS, similarity, syllabus

ExamType = Literal["GATE", "MIDSEM", "ENDSEM", "PRACTICE"]


class ExamRequest(BaseModel):
    area: str = Field(description="syllabus key, e.g. gate_cs, gate_da, mathematics_ii")
    exam: ExamType = "GATE"
    units: list[str] = Field(default_factory=list, description="focus units (exact unit names); empty = whole syllabus")
    topics: list[str] = Field(default_factory=list, description="narrower focus: exact topic names from the syllabus, if the student named specific topics")
    num_questions: int | None = Field(None, description="only for PRACTICE sets")
    difficulty: Literal["easy", "medium", "hard"] = "medium"


# ---------- 1. understanding the student's request ----------
ALIASES = {
    "Operating System": r"\bos\b|operating system",
    "Databases": r"dbms|database",
    "Database Management and Warehousing": r"dbms|database|warehous",
    "Computer Networks": r"\bcn\b|network",
    "Theory of Computation": r"\btoc\b|theory of computation|automata",
    "Compiler Design": r"compiler",
    "Computer Organization and Architecture": r"\bcoa\b|computer organi[sz]ation|architecture",
    "Programming and Data Structures": r"\bds\b|data structure|programming",
    "Programming, Data Structures and Algorithms": r"\bdsa\b|data structure|programming|python",
    "Algorithms": r"algorithm|\bdaa\b",
    "Digital Logic": r"digital",
    "Discrete Mathematics": r"discrete",
    "Linear Algebra": r"linear algebra",
    "Calculus": r"calculus",
    "Calculus and Optimization": r"calculus|optimi[sz]ation",
    "Probability and Statistics": r"probability|statistic",
    "Machine Learning": r"\bml\b|machine learning",
    "Artificial Intelligence": r"\bai\b(?! ?&)|artificial intelligence",
    "General Aptitude": r"aptitude|\bga\b",
}


def parse_request_offline(text: str) -> ExamRequest:
    t = text.lower()
    area = "gate_da" if re.search(r"\bda\b|data science", t) else "mathematics_ii" if re.search(r"math(ematics)?[\s-]*(ii|2)\b", t) else "gate_cs"
    exam = "MIDSEM" if re.search(r"mid[\s-]?sem|mid[\s-]?term", t) else "ENDSEM" if re.search(r"end[\s-]?sem|end[\s-]?term|final", t) else \
        "PRACTICE" if re.search(r"practice|quiz|set of|\d+\s*questions", t) else "GATE"
    unit_names = [u["unit_name"] for u in syllabus(area)["units"]]
    units = [u for u in unit_names if re.search(ALIASES.get(u, re.escape(u.lower())), t)]
    n = re.search(r"(\d+)\s*(questions|qs|mcqs?)", t)
    diff = "hard" if "hard" in t or "tough" in t else "easy" if "easy" in t else "medium"
    if units and exam == "GATE" and not re.search(r"full|mock|complete", t):
        exam = "PRACTICE"
    return ExamRequest(area=area, exam=exam, units=units, num_questions=int(n.group(1)) if n else None, difficulty=diff)


def parse_request(text: str, areas: list[str]) -> ExamRequest:
    if not llm.available():
        return parse_request_offline(text)
    catalogue = "\n".join(f"{a}:\n" + "\n".join(f"  {u['unit_name']}: " + " | ".join(u["topics"]) for u in syllabus(a)["units"]) for a in areas)
    try:
        req = llm.generate(
            f"A student asked: \"{text}\"\n\nAvailable syllabi (key: units):\n{catalogue}\n\n"
            "Map the request. University subjects (e.g. 'Operating Systems mid-sem') map to the GATE syllabus unit "
            "with that name unless a dedicated syllabus key exists. exam: GATE = full GATE mock, PRACTICE = short "
            "topic-wise set, MIDSEM/ENDSEM = university-style descriptive paper. Copy unit and topic names exactly; "
            "fill topics only when the student names specific topics (e.g. 'deadlock').",
            ExamRequest, temperature=0, think="MINIMAL", bulk=True,
        )
        valid = {u["unit_name"] for u in syllabus(req.area)["units"]} if req.area in areas else set()
        if not valid:
            return parse_request_offline(text)
        req.units = [u for u in req.units if u in valid]
        all_topics = {t for u in syllabus(req.area)["units"] for t in u["topics"]}
        req.topics = [t for t in req.topics if t in all_topics]
        return req
    except Exception:
        return parse_request_offline(text)


# ---------- 2. blueprint: the slots a paper must fill ----------
def blueprint(req: ExamRequest, rows: list[dict]) -> dict:
    """Return {title, duration_min, sections: [{name, instructions, slots: [{marks, qtype, section}]}]}."""
    subj = syllabus(req.area)
    if req.exam == "GATE":
        pattern = analysis.gate_pattern([r for r in rows if r["exam"] == "GATE"]) or []
        ga = [("GA", 1.0, "MCQ", 5), ("GA", 2.0, "MCQ", 5)]
        sub_sec = "DA" if req.area == "gate_da" else "CS"
        sub = []
        for marks, n_total in ((1.0, 25), (2.0, 30)):
            mix = {p["qtype"]: p["avg_count"] for p in pattern if p["section"] != "GA" and p["marks"] == marks}
            mix = mix or {"MCQ": 0.6 * n_total, "MSQ": 0.15 * n_total, "NAT": 0.25 * n_total}
            counts = _round_to_total(mix, n_total)
            sub += [(sub_sec, marks, t, c) for t, c in counts.items()]
        mk = lambda spec: [{"section": s, "marks": m, "qtype": t} for s, m, t, c in spec for _ in range(c)]  # noqa: E731
        return {
            "title": f"GATE Mock Test — {subj['subject'].replace('GATE ', '')}",
            "duration_min": 180, "total_marks": 100,
            "sections": [
                {"name": "General Aptitude", "instructions": "Q.1–Q.5 carry one mark each; Q.6–Q.10 carry two marks each.", "slots": mk(ga)},
                {"name": subj["subject"].replace("GATE ", ""), "instructions": "Q.11–Q.35 carry one mark each; Q.36–Q.65 carry two marks each. "
                 "MCQ: −1/3 (1-mark) or −2/3 (2-mark) for a wrong answer. No negative marking for MSQ and NAT.", "slots": mk(sub)},
            ],
        }
    if req.exam == "PRACTICE":
        n = req.num_questions or 20
        n1 = n // 2
        types = ["MCQ", "MCQ", "NAT", "MSQ"]
        slots = [{"section": "P", "marks": 1.0 if i < n1 else 2.0, "qtype": types[i % 4]} for i in range(n)]
        return {"title": f"Practice Set — {', '.join(req.topics or req.units) or subj['subject']}", "duration_min": 3 * n, "total_marks": sum(s["marks"] for s in slots),
                "sections": [{"name": "Practice", "instructions": "GATE-style questions. MCQ: one correct; MSQ: one or more correct; NAT: numerical answer.", "slots": slots}]}
    # ponytail: fixed common university patterns; infer from uploaded university papers when enough exist
    if req.exam == "MIDSEM":
        spec = [("Section A", "Answer ALL questions.", 5, 2.0, None), ("Section B", "Answer any FOUR questions.", 5, 5.0, 4)]
        title, dur = "Mid-Semester Examination", 90
    else:
        spec = [("Section A", "Answer ALL questions.", 10, 2.0, None), ("Section B", "Answer any SIX questions.", 8, 5.0, 6),
                ("Section C", "Answer any FIVE questions.", 7, 10.0, 5)]
        title, dur = "End-Semester Examination", 180
    focus = ", ".join(req.units) if req.units else subj["subject"]
    return {
        "title": f"{title} — {focus}", "duration_min": dur,
        "total_marks": sum(m * (a or n) for _, _, n, m, a in spec),
        "sections": [{"name": name, "instructions": ins, "attempt": a,
                      "slots": [{"section": name, "marks": m, "qtype": "DESCRIPTIVE"} for _ in range(n)]}
                     for name, ins, n, m, a in spec],
    }


def _round_to_total(mix: dict, total: int) -> dict:
    s = sum(mix.values()) or 1
    raw = {k: v * total / s for k, v in mix.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: total - sum(out.values())]:
        out[k] += 1
    return out


# ---------- 3. plan: which topic each slot tests ----------
def scope(req: ExamRequest) -> list[tuple[str, str]]:
    """(unit, topic) pairs in scope. Mid-sem covers the first half of each unit's topics."""
    units = [u for u in syllabus(req.area)["units"] if u["unit_number"] != 0]
    if req.units:
        units = [u for u in units if u["unit_name"] in req.units] or units
    pairs = []
    for u in units:
        ts = u["topics"][: math.ceil(len(u["topics"]) / 2)] if req.exam == "MIDSEM" else u["topics"]
        if req.exam == "MIDSEM" and not req.units and len(units) > 2:  # whole-course mid-sem: first half of units
            if u not in units[: math.ceil(len(units) / 2)]:
                continue
            ts = u["topics"]
        pairs += [(u["unit_name"], t) for t in ts]
    if req.topics:
        focused = [(u, t) for u in syllabus(req.area)["units"] for t in u["topics"] if t in req.topics]
        return [(u["unit_name"], t) for u, t in focused] or pairs
    return pairs


def plan(req: ExamRequest, bp: dict, stats: dict, seed: int | None = None) -> None:
    """Attach unit/topic/bloom to every slot by sampling topics in proportion to historical hotness."""
    rng = random.Random(seed)
    hot = {(t["unit"], t["topic"]): t["hot_score"] for t in stats.get("topics", [])}
    ga_topics = [(u["unit_name"], t) for u in syllabus(req.area)["units"] if u["unit_number"] == 0 for t in u["topics"]]
    pool = scope(req)
    weights = {p: hot.get(p, 0) + 10 for p in pool}  # +10: unseen syllabus topics still appear sometimes
    blooms = stats.get("blooms_pct") or {"Apply": 40, "Analyze": 30, "Understand": 20, "Remember": 10}
    for sec in bp["sections"]:
        for s in sec["slots"]:
            if s["section"] == "GA":
                s["unit"], s["topic"] = rng.choice(ga_topics)
            else:
                keys = list(weights)
                s["unit"], s["topic"] = rng.choices(keys, [weights[k] for k in keys])[0]
                weights[(s["unit"], s["topic"])] *= 0.5  # spread coverage
            if s["qtype"] == "DESCRIPTIVE" and s["marks"] >= 10:
                s["bloom"] = rng.choice(["Apply", "Analyze", "Evaluate", "Create"])
            else:
                s["bloom"] = rng.choices(list(blooms), list(blooms.values()))[0]


# ---------- 4. retrieval + generation ----------
class GenQuestion(BaseModel):
    slot: int
    text: str = Field(description="question stem; LaTeX in $...$; for DESCRIPTIVE may contain (a)/(b) sub-parts with marks")
    options: list[str] = Field(default_factory=list, description="exactly 4 for MCQ/MSQ as '(A) ...'; empty otherwise")
    answer: str = Field(description="MCQ: 'B'; MSQ: 'A;C'; NAT: '12' or '2.40 to 2.45'; DESCRIPTIVE: concise model answer / marking scheme")
    explanation: str = Field(description="short worked solution")


class GenBatch(BaseModel):
    questions: list[GenQuestion]


SYSTEM = (
    "You are a senior paper-setter for {exam_desc}. You write original, unambiguous, exam-standard questions "
    "with a single verifiable answer key. Every numeric answer must be computed carefully and double-checked in the explanation. "
    "Never copy reference questions; test the same concept with new data, framing or twist."
)


def exemplars(slot: dict, rows: list[dict], k: int = 3) -> list[str]:
    same = [r for r in rows if r["topic"] == slot["topic"]]
    same.sort(key=lambda r: (r["qtype"] != slot["qtype"], -(r["year"] or 0)))
    return [f"[{r['title']}, {r['qtype']}, {r['marks']:g}m] {r['text'][:500]} {' '.join(r['options'])[:200]}" for r in same[:k]]


def _llm_fill(req: ExamRequest, slots: list[dict], rows: list[dict], avoid_note: str = "") -> dict[int, GenQuestion]:
    exam_desc = "GATE (IIT/IISc national exam)" if req.exam in ("GATE", "PRACTICE") else "a university B.Tech examination"
    spec = []
    for s in slots:
        ex = "\n      ".join(exemplars(s, rows)) or "(none on record)"
        spec.append(f"slot {s['i']}: {s['qtype']}, {s['marks']:g} mark(s), unit '{s['unit']}', topic '{s['topic']}', "
                    f"Bloom level {s['bloom']}, difficulty {req.difficulty}\n   past questions on this topic:\n      {ex}")
    rules = (
        "MCQ: exactly 4 options, exactly one correct, answer is the letter. MSQ: 4 options, one or more correct, answer like 'A;C'. "
        "NAT: no options, numeric answer (a range 'x to y' when rounding is involved), state the rounding in the stem. "
        "DESCRIPTIVE: university long-answer style; answer = marking scheme with marks per step."
    )
    res = llm.generate(
        f"Write one new question for each slot below.\n{rules}\n{avoid_note}\n\n" + "\n\n".join(spec),
        GenBatch, system=SYSTEM.format(exam_desc=exam_desc), temperature=0.9,
    )
    return {q.slot: q for q in res.questions}


def _check(q: GenQuestion, qtype: str) -> list[str]:
    issues = []
    if qtype in ("MCQ", "MSQ"):
        if len(q.options) != 4:
            issues.append("needs exactly 4 options")
        letters = [a.strip() for a in q.answer.split(";")]
        if not letters or not all(l in "ABCD" and l for l in letters):
            issues.append("answer must be option letters")
        if qtype == "MCQ" and len(letters) != 1:
            issues.append("MCQ must have one answer")
    elif qtype == "NAT":
        if q.options:
            issues.append("NAT must not have options")
        if not re.fullmatch(r"-?\d+(\.\d+)?(\s*to\s*-?\d+(\.\d+)?)?", q.answer.strip()):
            issues.append("NAT answer must be numeric")
    return issues


# ---------- 5. verify answer keys (independent solve) ----------
class _Solve(BaseModel):
    slot: int
    answer: str = Field(description="MCQ: one letter; MSQ: letters joined by ';'; NAT: a number")
    well_posed: bool = Field(description="false if no option / more than one option fits an MCQ, or the data is insufficient")


class _Solves(BaseModel):
    solutions: list[_Solve]


def _same(a: str, b: str, qtype: str) -> bool:
    if qtype == "NAT":
        try:
            lo, _, hi = b.partition(" to ")
            x, lo = float(a.split(" to ")[0]), float(lo)
            hi = float(hi) if hi else lo
            tol = max(abs(lo), abs(hi), 1) * 0.01
            return lo - tol <= x <= hi + tol
        except ValueError:
            return False
    norm = lambda v: sorted(c for c in v.upper() if c in "ABCD")  # noqa: E731
    return norm(a) == norm(b)


def verify(items: list[tuple[int, dict, str]]) -> set[int]:
    """Solve each objective question afresh, blind to the key. Returns slots whose key is wrong or ill-posed."""
    if not items:
        return set()
    listing = "\n\n".join(f"slot {i} ({qt}): {q['text']}\n" + "\n".join(q.get("options") or []) for i, q, qt in items)
    res = llm.generate("Solve each question carefully, step by step in your head, then give only the final answer. "
                       "Flag well_posed=false if the question is flawed.\n\n" + listing, _Solves, temperature=0, think="HIGH")
    got = {s.slot: s for s in res.solutions}
    return {i for i, q, qt in items if (s := got.get(i)) is None or not s.well_posed or not _same(s.answer, q["answer"], qt)}


# ---------- 6. assemble ----------
def generate_paper(req: ExamRequest, rows: list[dict], seed: int | None = None, novelty_threshold: float = 0.8) -> dict:
    stats = analysis.analyze(rows) if rows else {}
    bp = blueprint(req, rows)
    plan(req, bp, stats, seed)
    slots = [s for sec in bp["sections"] for s in sec["slots"]]
    for i, s in enumerate(slots, 1):
        s["i"] = i
    corpus = [_full(r) for r in rows]
    use_llm = llm.available()
    filled: dict[int, dict] = {}

    if use_llm:
        by_unit: dict[str, list[dict]] = {}
        for s in slots:
            by_unit.setdefault(s["unit"], []).append(s)
        batches = [b[i : i + 6] for b in by_unit.values() for i in range(0, len(b), 6)]
        def run(batch):
            try:
                return _llm_fill(req, batch, rows)
            except Exception as e:
                print(f"generation batch failed ({e}); using past questions for it")
                return {}

        with ThreadPoolExecutor(6) as pool:
            results = list(pool.map(run, batches))
        for batch, got in zip(batches, results):
            for s in batch:
                if (q := got.get(s["i"])) and not _check(q, s["qtype"]):
                    filled[s["i"]] = q.model_dump() | {"source": "generated"}
        # novelty: regenerate anything too close to a real past question
        if corpus and filled:
            ids = list(filled)
            S = similarity([_full(filled[i]) for i in ids], corpus)
            redo = [i for i, row in zip(ids, S) if row.max() >= novelty_threshold]
            if redo:
                note = "These drafts were too close to past papers; write genuinely different questions:\n" + \
                       "\n".join(f"- {filled[i]['text'][:200]}" for i in redo)
                try:
                    got = _llm_fill(req, [s for s in slots if s["i"] in redo], rows, note)
                    for i in redo:
                        if (q := got.get(i)) and not _check(q, slots[i - 1]["qtype"]):
                            filled[i] = q.model_dump() | {"source": "generated"}
                except Exception:
                    pass

    # independent re-solve of objective questions; wrong/ill-posed keys get one regeneration, then a past question
    if use_llm and filled:
        objective = {s["i"]: s for s in slots if s["qtype"] in ("MCQ", "MSQ", "NAT") and s["i"] in filled}

        def verify_batch(ids):
            try:
                return verify([(i, filled[i], objective[i]["qtype"]) for i in ids])
            except Exception as e:
                print(f"verification batch failed ({e})")
                return set()

        ids = list(objective)
        with ThreadPoolExecutor(6) as pool:
            bad = set().union(*pool.map(verify_batch, [ids[k : k + 8] for k in range(0, len(ids), 8)])) if ids else set()
        if bad:
            try:
                got = _llm_fill(req, [objective[i] for i in bad], rows,
                                "Previous drafts had wrong or ambiguous answer keys; make sure exactly the keyed answer is correct.")
                fresh = {i: q.model_dump() | {"source": "generated"} for i, q in got.items() if i in bad and not _check(q, objective[i]["qtype"])}
                still = verify_batch(list(fresh)) if fresh else set()
                for i in bad:
                    if i in fresh and i not in still:
                        filled[i] = fresh[i]
                    else:
                        filled.pop(i, None)  # falls back to a real past question below
            except Exception:
                for i in bad:
                    filled.pop(i, None)
        verified_fixed = len(bad)
    else:
        verified_fixed = 0

    # fill gaps (or everything, offline) with real past questions matching the slot
    used: set[int] = set()
    for s in slots:
        if s["i"] in filled:
            continue
        if pyq := _pick_pyq(s, rows, used):
            used.add(pyq["id"])
            filled[s["i"]] = {"text": pyq["text"], "options": pyq["options"], "answer": pyq["answer"] or "",
                              "explanation": "", "source": "pyq", "ref": f"{pyq['title']} Q{pyq['qno']}"}

    sections = []
    for sec in bp["sections"]:
        qs = []
        for s in sec["slots"]:
            q = filled.get(s["i"])
            if not q:
                continue
            qs.append({"n": s["i"], "marks": s["marks"], "qtype": s["qtype"], "unit": s["unit"], "topic": s["topic"], "bloom": s["bloom"],
                       **{k: q.get(k) for k in ("text", "options", "answer", "explanation", "source", "ref")}})
        sections.append({"name": sec["name"], "instructions": sec["instructions"], "attempt": sec.get("attempt"), "questions": qs})

    paper = {
        "title": bp["title"], "exam": req.exam, "area": req.area, "duration_min": bp["duration_min"], "total_marks": bp["total_marks"],
        "request": req.model_dump(), "generated_by": config.GEMINI_MODEL if use_llm else "offline (past questions)",
        "sections": sections,
    }
    paper["validation"] = validate(paper, bp, corpus, novelty_threshold) | {"keys_failed_verification": verified_fixed}
    return paper


def _full(q: dict) -> str:
    """Stem + options: many GATE stems are generic ('Which of the following is/are TRUE?')."""
    return f"{q['text']} {' '.join(q.get('options') or [])}"


def _pick_pyq(slot: dict, rows: list[dict], used: set) -> dict | None:
    cands = [r for r in rows if r["id"] not in used and (slot["section"] == "GA") == (r["section"] == "GA")]
    score = lambda r: (r["topic"] == slot["topic"]) * 4 + (r["unit"] == slot["unit"]) * 2 + (r["qtype"] == slot["qtype"]) + (r["marks"] == slot["marks"])  # noqa: E731
    best = max(cands, key=lambda r: (score(r), r["year"] or 0), default=None)
    return best if best and score(best) >= 2 else None


def validate(paper: dict, bp: dict, corpus: list[str], threshold: float) -> dict:
    qs = [q for s in paper["sections"] for q in s["questions"]]
    expected = sum(len(s["slots"]) for s in bp["sections"])
    marks = sum(q["marks"] * 1 for s, sec in zip(paper["sections"], bp["sections"]) for q in s["questions"][: sec.get("attempt") or len(s["questions"])])
    gen = [q for q in qs if q["source"] == "generated"]
    max_sim = similarity([_full(q) for q in gen], corpus).max(axis=1).round(2).tolist() if gen and corpus else []
    units = {}
    for q in qs:
        units[q["unit"]] = units.get(q["unit"], 0) + q["marks"]
    bloom_mix = {b: sum(1 for q in qs if q["bloom"] == b) for b in BLOOM_LEVELS}
    return {
        "questions": f"{len(qs)}/{expected}",
        "marks_ok": abs(marks - paper["total_marks"]) < 1e-6,
        "attemptable_marks": marks,
        "generated": len(gen),
        "from_past_papers": len(qs) - len(gen),
        "max_similarity_to_past": max(max_sim, default=None),
        "too_similar": sum(s >= threshold for s in max_sim),
        "unit_marks": units,
        "bloom_mix": {k: v for k, v in bloom_mix.items() if v},
    }
