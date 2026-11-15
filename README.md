# PaperMind — past-paper analytics (NLP) + exam paper generation (GenAI)

A student asks *"Give me a mid-sem paper for Operating Systems"* or *"GATE DA full mock"*, and PaperMind
generates a fresh paper grounded in what past papers actually asked. The corpus is official GATE CS & DA papers
plus any university papers you upload.

## Pipeline

```
 PDFs ─▶ ingest.py ─▶ nlp.py ─────────────────────────▶ SQLite (data/papermind.db)
         GATE: regex + official key      Bloom's level (lexicon rules)
         (type, marks, answer)           topic: Gemini labels ▸ zero-shot embeddings
         university: heuristics /        └─▶ distilled into TF-IDF + LogReg (runtime model)
         Gemini vision for maths/scans
                                            analysis.py: unit weightage, trends, hot topics,
                                            MCQ/MSQ/NAT mix, repeated concepts
 request ─▶ generate.py: parse (Gemini/regex) ▸ blueprint (from history) ▸ plan topics by hot-score
           ▸ retrieve past questions (RAG) ▸ Gemini writes ▸ validate (format, novelty, marks)
           ▸ verify keys: a second blind Gemini solve; mismatches regenerated, else replaced by a real PYQ
           offline (no key): same plan filled with real past questions
 student ─▶ assistant.py: Gemini agent with tools = the functions above + progress tracking
           (syllabus_overview, hot_topics, search_past_questions, create_practice_set, get_answer,
            record_result, my_progress)  → priority = exam weight × student weakness
```

| File | Role |
|---|---|
| `papermind/ingest.py` | PDF → questions (GATE parser + answer-key join, university segmenter, Gemini vision) |
| `papermind/nlp.py` | Bloom's taxonomy, syllabus topic labelling, classifier training/eval, similarity |
| `papermind/analysis.py` | historical statistics |
| `papermind/generate.py` | request parsing, blueprint, planning, RAG generation, validation |
| `papermind/assistant.py` | personal study assistant (Gemini function calling over the modules above) |
| `papermind/app.py` / `cli.py` | REST API / command line |
| `papermind/data/syllabus/*.json` | official GATE CS & DA syllabi, Mathematics-II |

## Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
cp .env.example .env            # add GEMINI_API_KEY (free: https://aistudio.google.com/apikey)
```

## Build the corpus

Put papers in `data/gate/raw/` named `GATE_<year>_<CS1|CS2|DA>.pdf`, with the official key as
`GATE_<year>_<CS1|CS2|DA>_key.pdf` (optional, but it supplies the exact type, marks and answer). Then:

```bash
.venv/bin/python -m papermind.cli ingest-gate data/gate/raw    # labels with Gemini if a key is set
.venv/bin/python -m papermind.cli train                        # held-out-year accuracy is printed
.venv/bin/python -m papermind.cli ingest-paper data/samples/Mathematics_II_endsem_2025.pdf --area mathematics_ii --exam ENDSEM --year 2025
```

### Adding new papers (any layout)

`ingest-paper` (CLI) and `POST /api/papers` (API, multipart `files` + `area`, `exam`, `year`) accept a digital PDF,
a scanned PDF, or photos of the pages in order:

```bash
.venv/bin/python -m papermind.cli ingest-paper page1.jpg page2.jpg page3.jpg --area gate_cs --exam GATE --year 2027
```

1. **Text** — PDF text layer; if missing or garbled → local OCR (macOS Vision via `ocrmac`); if OCR is unavailable
   (e.g. the Linux server) → Gemini vision.
2. **Structure** — the GATE parser when the paper fits the GATE layout; otherwise Gemini splits the text into
   questions and reports every question format it saw, so a changed exam pattern is still captured. Offline, a
   heuristic segmenter is used.
3. **Labels** — unit/topic (Gemini → trained model → zero-shot) and Bloom's level, then stored.

Gemini's free tier caps requests per model per day, so `.env` lists model chains: bulk work (OCR structuring,
labelling) uses flash-lite models; paper generation uses the strongest available. When one model's daily quota is
used up, the next is tried automatically.

## Use

```bash
.venv/bin/python -m papermind.cli analyze gate_cs
.venv/bin/python -m papermind.cli generate "mid-sem paper for Operating Systems" --seed 1
.venv/bin/python -m papermind.cli chat --student riya  # personal assistant: /progress, /reset, exit
.venv/bin/uvicorn papermind.app:app --reload        # API docs at http://localhost:8000/docs
```

Example assistant session: *"Which GATE CS subjects carry the most marks?"* → answers from 16 years of papers;
*"Quiz me on deadlock"* → generates a verified set, asks one question at a time, checks your answer against the key,
records it; *"Make me a 4-week plan"* → orders topics by exam weight × your measured weakness.

Endpoints: `GET /api/health`, `GET|POST /api/papers`, `DELETE /api/papers/{id}`, `GET /api/questions`,
`GET /api/analysis/{area}`, `POST /api/generate {"request": "..."}`, `GET /api/generated[/{id}]`,
`POST /api/chat {"student", "message"}`, `DELETE /api/chat?student=`, `GET /api/progress?student=`.

## Deploy (free) — Render

Render's free web service builds this repo's `Dockerfile` (512 MB RAM, sleeps after 15 min idle, ~30–60 s cold
start, no persistent disk: uploads/chats/progress reset on restart; the prebuilt corpus ships in the image).
Step-by-step: push to GitHub → render.com → New → Web Service → connect the repo → Language **Docker**,
Instance type **Free** → add env var `GEMINI_API_KEY` → Deploy.

## Tests

`.venv/bin/python -m pytest -q tests`  (offline, no API key needed)

The previous version of the codebase is archived in `legacy_v1_backup.tar.gz`.
