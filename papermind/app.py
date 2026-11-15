"""HTTP API (FastAPI). Frontend and assistant come later and talk to these endpoints."""
import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from papermind import db, generate, ingest, llm, nlp
from papermind.analysis import analyze

app = FastAPI(title="PaperMind", description="Past-paper analytics (NLP) + exam paper generation (GenAI)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
con = db.connect()


@app.get("/api/health")
def health():
    return {"ok": True, "llm": llm.available(), "areas": nlp.areas()}


@app.get("/api/papers")
def papers():
    return [dict(r) for r in con.execute(
        "SELECT p.*, COUNT(q.id) AS questions FROM papers p LEFT JOIN questions q ON q.paper_id = p.id GROUP BY p.id ORDER BY area, year")]


@app.post("/api/papers")
def upload(files: list[UploadFile] = File(..., description="one PDF (digital or scanned) or page photos in order"),
           area: str = Form(...), exam: str = Form("ENDSEM"), year: int | None = Form(None), title: str | None = Form(None)):
    if area not in nlp.areas():
        raise HTTPException(400, f"unknown area; choose one of {nlp.areas()}")
    names = [Path(f.filename or "").name for f in files]
    if not all(Path(n).suffix.lower() in {".pdf"} | ingest.IMAGE_EXT for n in names):
        raise HTTPException(400, "upload a PDF or images (png/jpg/webp/heic)")
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for f, n in zip(files, names):
            paths.append(Path(d) / n)
            with paths[-1].open("wb") as out:
                shutil.copyfileobj(f.file, out)
        try:
            if len(paths) == 1 and ingest.GATE_NAME.search(names[0]):
                paper = ingest.ingest_gate(paths[0])
            else:
                paper = ingest.ingest_any(paths if len(paths) > 1 else paths[0], area, exam, year, title)
        except (ValueError, llm.LLMUnavailable) as e:
            raise HTTPException(422, str(e))
    nlp.label(paper["questions"], paper["area"])
    pid = db.add_paper(con, paper["exam"], paper["area"], paper["year"], paper["title"], title or names[0], paper["questions"])
    return {"id": pid, "questions": len(paper["questions"]), "method": paper.get("method", "GATE parser")}


@app.delete("/api/papers/{pid}")
def delete(pid: int):
    con.execute("DELETE FROM papers WHERE id = ?", (pid,))
    con.commit()
    return {"deleted": pid}


@app.get("/api/questions")
def questions(area: str, unit: str | None = None, topic: str | None = None, limit: int = 100):
    rows = [r for r in db.questions(con, area) if (not unit or r["unit"] == unit) and (not topic or r["topic"] == topic)]
    return rows[-limit:]


_analysis_cache: dict = {}


@app.get("/api/analysis/{area}")
def analysis(area: str, unit: str | None = None):
    # cached per corpus state: the free host's CPU makes the repeat-concept scan slow
    n = con.execute("SELECT COUNT(*), MAX(q.id) FROM questions q JOIN papers p ON p.id = q.paper_id WHERE p.area = ?", (area,)).fetchone()
    key = (area, unit, tuple(n))
    if key not in _analysis_cache:
        rows = db.questions(con, area)
        if unit:
            rows = [r for r in rows if r["unit"] == unit]
        _analysis_cache[key] = json.loads(json.dumps(analyze(rows), default=list))
    return _analysis_cache[key]


class GenerateIn(BaseModel):
    request: str  # e.g. "Give me a mid-sem paper for Operating Systems"
    seed: int | None = None


@app.post("/api/generate")
def generate_paper(body: GenerateIn):
    req = generate.parse_request(body.request, nlp.areas())
    paper = generate.generate_paper(req, db.questions(con, req.area), seed=body.seed)
    paper["id"] = con.execute("INSERT INTO generated (request, paper) VALUES (?, ?)", (body.request, json.dumps(paper))).lastrowid
    con.commit()
    return paper


@app.get("/api/generated")
def generated(limit: int = 20):
    return [{"id": r["id"], "request": r["request"], "created_at": r["created_at"]}
            for r in con.execute("SELECT id, request, created_at FROM generated ORDER BY id DESC LIMIT ?", (limit,))]


@app.get("/api/generated/{gid}")
def generated_one(gid: int):
    row = con.execute("SELECT paper FROM generated WHERE id = ?", (gid,)).fetchone()
    if not row:
        raise HTTPException(404)
    return json.loads(row["paper"])


class ChatIn(BaseModel):
    student: str = "me"
    message: str


@app.post("/api/chat")
def chat(body: ChatIn):
    from papermind import assistant

    try:
        return {"reply": assistant.chat(con, body.student, body.message)}
    except llm.LLMUnavailable as e:
        raise HTTPException(503, str(e))


@app.get("/api/progress")
def progress(student: str = "me", area: str = ""):
    from papermind import assistant

    return assistant.progress(con, student, area)


@app.get("/api/chat")
def chat_history(student: str = "me"):
    """Plain user/model text turns (tool calls omitted) for redisplaying a conversation."""
    row = con.execute("SELECT history FROM chats WHERE student = ?", (student,)).fetchone()
    out = []
    for c in json.loads(row["history"]) if row else []:
        text = "".join(p.get("text", "") for p in c.get("parts", []) if not p.get("thought"))
        if text.strip() and c.get("role") in ("user", "model"):
            out.append({"role": c["role"], "text": text})
    return out


@app.delete("/api/chat")
def reset_chat(student: str = "me"):
    from papermind import assistant

    assistant.reset(con, student)
    return {"reset": student}


INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX)
