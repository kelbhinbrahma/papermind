"""Historical analysis of past papers: what gets asked, how much it weighs, where it is heading."""
from collections import Counter, defaultdict

import numpy as np

from papermind.nlp import similarity

DECAY = 0.8  # weight of a paper one year older than the latest (recency bias for "hot" topics)


def _slope(series: list[float]) -> float:
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series))
    return float(np.polyfit(x, series, 1)[0])


def analyze(rows: list[dict]) -> dict:
    if not rows:
        return {"papers": 0}
    papers = {r["paper_id"]: (r["year"], r["title"]) for r in rows}
    n_papers = len(papers)
    years = sorted({y for y, _ in papers.values() if y})
    latest = years[-1] if years else 0
    papers_per_year = Counter(y for y, _ in papers.values())

    # unit (subject) weightage: average marks per paper, per year
    unit_year = defaultdict(lambda: defaultdict(float))
    topic = defaultdict(lambda: {"count": 0, "marks": 0.0, "years": set(), "hot": 0.0, "unit": None})
    for r in rows:
        m = r["marks"] or 0
        unit = r["unit"] or "Unlabelled"
        unit_year[unit][r["year"]] += m
        t = topic[(unit, r["topic"] or "Unlabelled")]
        t["count"] += 1
        t["marks"] += m
        t["years"].add(r["year"])
        t["unit"] = unit
        t["hot"] += m * DECAY ** (latest - (r["year"] or latest)) / max(papers_per_year[r["year"]], 1)

    total = sum(r["marks"] or 0 for r in rows) or 1
    units = []
    for u, by_year in unit_year.items():
        per_year = [by_year.get(y, 0) / papers_per_year[y] for y in years]
        s = _slope(per_year)
        units.append({
            "unit": u,
            "avg_marks_per_paper": round(sum(by_year.values()) / n_papers, 2),
            "share_pct": round(100 * sum(by_year.values()) / total, 1),
            "marks_by_year": {y: round(v, 2) for y, v in zip(years, per_year)},
            "trend": "rising" if s > 0.5 else "falling" if s < -0.5 else "stable",
        })
    units.sort(key=lambda u: -u["avg_marks_per_paper"])

    max_hot = max(t["hot"] for t in topic.values()) or 1
    topics = sorted(
        (
            {
                "unit": t["unit"], "topic": name[1], "questions": t["count"],
                "avg_marks_per_paper": round(t["marks"] / n_papers, 2),
                "years_seen": len(t["years"]), "of_years": len(years), "last_seen": max(t["years"]),
                "hot_score": round(100 * t["hot"] / max_hot, 1),
            }
            for name, t in topic.items()
        ),
        key=lambda t: -t["hot_score"],
    )

    qtype = Counter(r["qtype"] for r in rows)
    marks_split = Counter(r["marks"] for r in rows)
    blooms = Counter(r["bloom"] for r in rows)

    return {
        "papers": n_papers,
        "years": years,
        "questions": len(rows),
        "units": units,
        "topics": topics,
        "question_types": {k: round(v / n_papers, 1) for k, v in qtype.items()},
        "marks_split": {str(k): round(v / n_papers, 1) for k, v in marks_split.items()},
        "blooms_pct": {k: round(100 * v / len(rows), 1) for k, v in blooms.most_common()},
        "figure_questions_pct": round(100 * sum(r["has_figure"] for r in rows) / len(rows), 1),
        "repeated_concepts": repeated(rows),
    }


def repeated(rows: list[dict], threshold: float = 0.55, top: int = 15) -> list[dict]:
    """Near-identical questions asked in different years (concept recurrence)."""
    rows = [r for r in rows if r.get("section") != "GA" and len(r["text"]) > 40]
    if len(rows) < 2:
        return []
    texts = [r["text"] for r in rows]
    S = similarity(texts, texts)
    pairs = []
    for i, j in zip(*np.where(np.triu(S, 1) >= threshold)):
        a, b = rows[i], rows[j]
        if a["year"] != b["year"]:
            pairs.append({"similarity": round(float(S[i, j]), 2), "a": f"{a['title']} Q{a['qno']}", "b": f"{b['title']} Q{b['qno']}",
                          "topic": a["topic"], "text": a["text"][:160]})
    return sorted(pairs, key=lambda p: -p["similarity"])[:top]


def gate_pattern(rows: list[dict]) -> list[dict]:
    """Average GATE slot layout per paper: (section, marks, qtype) -> count. Drives the mock blueprint."""
    n = len({r["paper_id"] for r in rows}) or 1
    c = Counter((r["section"], r["marks"], r["qtype"]) for r in rows)
    return [{"section": s, "marks": m, "qtype": t, "avg_count": round(v / n, 2)} for (s, m, t), v in sorted(c.items(), key=str)]
