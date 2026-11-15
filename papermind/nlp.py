"""
NLP layer: Bloom's taxonomy, syllabus topic labelling, a trainable topic classifier, similarity.

Labelling strategy (best available wins):
  1. Gemini reads each question against the official syllabus      -> label_source='llm'
  2. zero-shot: sentence-embedding similarity to syllabus topics   -> label_source='zero-shot'
Then `train()` distils those labels into a TF-IDF + logistic-regression model that is tiny
enough to run on the deployed server for any newly uploaded paper  -> label_source='model'
"""
import difflib
import json
import re
from functools import lru_cache

import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from papermind import config, llm

# ---------- Bloom's taxonomy ----------
BLOOM_LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
_BLOOM_CUES = {
    "Create": r"design|devise|formulate|construct an? (algorithm|grammar|automaton|dfa|nfa)|write an? (program|function|query)|propose",
    "Evaluate": r"justify|evaluate|critique|which (of the following )?is (better|best|most appropriate)|assess|recommend",
    "Analyze": r"prove|derive|analy[sz]e|compare|differentiate|distinguish|time complexity|worst[- ]case|which (one )?of the following( [\w()]+){0,2} (is|are|is/are) (true|false|correct|incorrect|valid)",
    "Apply": r"calculate|compute|find|determine|evaluate the|solve|number of|how many|value of|output of|minimum|maximum|probability",
    "Understand": r"explain|describe|discuss|illustrate|summari[sz]e|interpret|what happens",
    "Remember": r"define|state|list|name|what is|which (one )?of the following (is|are)\b",
}


def bloom(text: str) -> str:
    """Highest-order cue that appears wins; lexicon rules are transparent and explainable."""
    t = text.lower()
    for level in reversed(BLOOM_LEVELS):
        if re.search(rf"\b({_BLOOM_CUES[level]})", t):
            return level
    return "Understand"


# ---------- syllabus ----------
@lru_cache
def syllabus(area: str) -> dict:
    return json.loads((config.SYLLABUS_DIR / f"{area}.json").read_text())


def areas() -> list[str]:
    return sorted(p.stem for p in config.SYLLABUS_DIR.glob("*.json"))


def topic_index(area: str) -> list[tuple[str, str]]:
    """Flat list of (unit_name, topic)."""
    return [(u["unit_name"], t) for u in syllabus(area)["units"] for t in u["topics"]]


def _units_for(q: dict, area: str) -> list[dict]:
    """GA questions only map to the General Aptitude unit and vice versa."""
    units = syllabus(area)["units"]
    if q.get("section") == "GA":
        return [u for u in units if u["unit_number"] == 0] or units
    return [u for u in units if u["unit_number"] != 0] or units


# ---------- 1. LLM labelling ----------
class _Label(BaseModel):
    qno: str
    unit: str
    topic: str


class _Labels(BaseModel):
    labels: list[_Label]


def label_with_llm(qs: list[dict], area: str) -> None:
    syl = "\n".join(f"- {u['unit_name']}: " + "; ".join(u["topics"]) for u in syllabus(area)["units"])
    for i in range(0, len(qs), 33):
        batch = qs[i : i + 33]
        listing = "\n\n".join(f"[{q['qno']}] ({q.get('section')}) {q['text'][:700]} {' '.join(q.get('options', []))[:300]}" for q in batch)
        res = llm.generate(
            f"Syllabus (unit: topics):\n{syl}\n\nFor each question, choose the single best unit and the closest "
            f"topic string copied exactly from the syllabus. Questions in section GA belong to General Aptitude.\n\n{listing}",
            _Labels, temperature=0, think="LOW", bulk=True,
        )
        index = topic_index(area)
        got = {l.qno: l for l in res.labels}
        for q in batch:
            if l := got.get(q["qno"]):  # snap near-miss strings to the closest real syllabus topic
                cands = [f"{u} || {t}" for u, t in index if u == l.unit] or [f"{u} || {t}" for u, t in index]
                best = difflib.get_close_matches(f"{l.unit} || {l.topic}", cands, n=1, cutoff=0.4)
                if best:
                    u, t = best[0].split(" || ")
                    q.update(unit=u, topic=t, label_source="llm")


# ---------- 2. zero-shot embeddings ----------
@lru_cache(maxsize=1)
def _encoder():
    from sentence_transformers import SentenceTransformer  # heavy; only needed at ingest time

    return SentenceTransformer("all-MiniLM-L6-v2")


def label_zero_shot(qs: list[dict], area: str) -> None:
    enc = _encoder()
    for group in ("GA", "subject"):
        sub = [q for q in qs if (q.get("section") == "GA") == (group == "GA") and not q.get("topic")]
        if not sub:
            continue
        units = _units_for(sub[0], area)
        cands = [(u["unit_name"], t) for u in units for t in u["topics"]]
        T = enc.encode([f"{u}: {t}" for u, t in cands], normalize_embeddings=True)
        Q = enc.encode([q["text"] + " " + " ".join(q.get("options", [])) for q in sub], normalize_embeddings=True)
        best = (Q @ T.T).argmax(1)
        for q, b in zip(sub, best):
            q.update(unit=cands[b][0], topic=cands[b][1], label_source="zero-shot")


def label(qs: list[dict], area: str, use_llm: bool | None = None, use_model: bool = True) -> list[dict]:
    """Fill unit/topic/bloom in place. The CLI corpus ingest passes use_model=False so the trained
    model never learns from its own predictions; the server uses it for new uploads."""
    for q in qs:
        q["bloom"] = bloom(q["text"])
    if use_llm is None:
        use_llm = llm.available()
    if use_llm:
        try:
            label_with_llm(qs, area)
        except Exception as e:  # rate limit / network: fall through to offline labelling
            print(f"LLM labelling failed ({e}); using zero-shot for the rest")
    if use_model and (model := load_model(area)) is not None:
        for q in (q for q in qs if not q.get("topic")):
            q["unit"], q["topic"] = model.predict([_feat(q)])[0].split(" || ")
            q["label_source"] = "model"
    if any(not q.get("topic") for q in qs):
        label_zero_shot(qs, area)
    return qs


# ---------- 3. supervised classifier (distilled from the labels above) ----------
def _feat(q: dict) -> str:
    return f"{q.get('section') or ''} {q['text']} {' '.join(q.get('options') or [])}"


def train(rows: list[dict], area: str) -> dict:
    """Train TF-IDF + LogisticRegression on labelled rows; hold out the latest year for evaluation."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline

    rows = [r for r in rows if r.get("topic") and r.get("label_source") in ("llm", "manual", "zero-shot")]
    y = [f"{r['unit']} || {r['topic']}" for r in rows]
    X = [_feat(r) for r in rows]
    years = sorted({r["year"] for r in rows})
    make = lambda: make_pipeline(  # noqa: E731
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
        LogisticRegression(max_iter=2000, C=5, class_weight="balanced"),
    )
    report = {"area": area, "n": len(rows), "classes": len(set(y)), "years": years}
    if len(years) > 1:
        test = [r["year"] == years[-1] for r in rows]
        tr = [i for i, t in enumerate(test) if not t]
        te = [i for i, t in enumerate(test) if t]
        m = make().fit([X[i] for i in tr], [y[i] for i in tr])
        pred = m.predict([X[i] for i in te])
        gold = [y[i] for i in te]
        report |= {
            "held_out_year": years[-1],
            "topic_accuracy": round(accuracy_score(gold, pred), 3),
            "topic_macro_f1": round(f1_score(gold, pred, average="macro", zero_division=0), 3),
            "unit_accuracy": round(np.mean([p.split(" || ")[0] == g.split(" || ")[0] for p, g in zip(pred, gold)]), 3),
        }
    model = make().fit(X, y)
    models = joblib.load(config.MODEL_PATH) if config.MODEL_PATH.exists() else {}
    models[area] = model
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, config.MODEL_PATH)
    load_model.cache_clear()
    return report


@lru_cache
def load_model(area: str):
    if not config.MODEL_PATH.exists():
        return None
    import joblib

    return joblib.load(config.MODEL_PATH).get(area)


# ---------- similarity (TF-IDF: light, deterministic, good on technical vocabulary) ----------
def similarity(a: list[str], b: list[str]) -> np.ndarray:
    """Cosine similarity matrix len(a) x len(b) using word + char n-grams."""
    if not a or not b:
        return np.zeros((len(a), len(b)))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(a + b)
    return cosine_similarity(vec.transform(a), vec.transform(b))
