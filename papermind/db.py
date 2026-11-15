"""SQLite storage. One file, stdlib only, ships with the deploy."""
import json
import sqlite3

from papermind.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY,
    exam TEXT NOT NULL,          -- 'GATE' | 'MIDSEM' | 'ENDSEM' | 'OTHER'
    area TEXT NOT NULL,          -- syllabus key: 'gate_cs', 'gate_da', 'mathematics_ii', ...
    year INTEGER,
    title TEXT,
    filename TEXT UNIQUE,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    qno TEXT,
    text TEXT NOT NULL,
    options TEXT,                -- JSON list, MCQ/MSQ only
    qtype TEXT,                  -- MCQ | MSQ | NAT | DESCRIPTIVE
    marks REAL,
    section TEXT,                -- GA | CS | DA | Part-I ...
    answer TEXT,
    unit TEXT,                   -- syllabus unit name (subject inside GATE)
    topic TEXT,
    label_source TEXT,           -- 'llm' | 'zero-shot' | 'model' | 'manual'
    bloom TEXT,
    has_figure INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chats (
    student TEXT PRIMARY KEY,
    history TEXT NOT NULL,       -- JSON list of Gemini Content (incl. tool calls)
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY,
    student TEXT NOT NULL,
    area TEXT, unit TEXT, topic TEXT,
    correct INTEGER NOT NULL,
    source TEXT,                 -- e.g. 'generated:12#Q3'
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS generated (
    id INTEGER PRIMARY KEY,
    request TEXT,
    paper TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def connect(path=None) -> sqlite3.Connection:
    p = path or DB_PATH
    if str(p) != ":memory:":
        p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


def add_paper(con, exam, area, year, title, filename, questions) -> int:
    """Insert a paper and its questions, replacing any earlier ingest of the same file."""
    con.execute("DELETE FROM papers WHERE filename = ?", (filename,))
    pid = con.execute(
        "INSERT INTO papers (exam, area, year, title, filename) VALUES (?,?,?,?,?)",
        (exam, area, year, title, filename),
    ).lastrowid
    con.executemany(
        """INSERT INTO questions (paper_id, qno, text, options, qtype, marks, section, answer,
           unit, topic, label_source, bloom, has_figure)
           VALUES (:paper_id, :qno, :text, :options, :qtype, :marks, :section, :answer,
           :unit, :topic, :label_source, :bloom, :has_figure)""",
        [
            {
                "paper_id": pid, "qno": None, "options": None, "qtype": None, "marks": None,
                "section": None, "answer": None, "unit": None, "topic": None,
                "label_source": None, "bloom": None, "has_figure": 0,
                **q, "options": json.dumps(q["options"]) if q.get("options") else None,
            }
            for q in questions
        ],
    )
    con.commit()
    return pid


def questions(con, area=None, exam=None) -> list[dict]:
    sql = """SELECT q.*, p.year, p.exam, p.area, p.title FROM questions q
             JOIN papers p ON p.id = q.paper_id WHERE 1=1"""
    args = []
    if area:
        sql += " AND p.area = ?"
        args.append(area)
    if exam:
        sql += " AND p.exam = ?"
        args.append(exam)
    rows = [dict(r) for r in con.execute(sql + " ORDER BY p.year, q.id", args)]
    for r in rows:
        r["options"] = json.loads(r["options"]) if r["options"] else []
    return rows
