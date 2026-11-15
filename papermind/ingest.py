"""
PDF -> structured questions.

GATE papers (digital PDFs from the organising IIT) are parsed offline with regex and joined to
the official answer key (type, marks, answer). University papers are segmented heuristically.
When a Gemini key is set, scanned pages and garbled maths/figures go through Gemini vision.
"""
import re
from pathlib import Path

import fitz  # PyMuPDF
from pydantic import BaseModel

from papermind import llm

NOISE = re.compile(
    r"^(organi[sz]ing institute.*|page \d+ of \d+|.*\((cs|da|cs-?\d)\)\s*$|general aptitude.*|"
    r"q\.\s*\d+\s*[–-]\s*q\.\s*\d+.*carry.*|computer science and information technology.*|"
    r"data science (and|&) artificial intelligence.*)$",
    re.I,
)
Q_START = re.compile(r"^Q\.\s*(\d{1,2})(?!\s*([–-]|to)\s*Q)\s*(.*)$")
OPTION = re.compile(r"^\(([A-D])\)\s*(.*)$")
FIGURE = re.compile(r"\b(figure|shown below|given below|diagram|as shown|following (tree|graph|circuit|table))\b", re.I)
GATE_NAME = re.compile(r"GATE_(\d{4})_(CS\d?|DA)(_sample\d*)?(?:_key)?\.pdf$", re.I)


def page_texts(pdf_path) -> list[str]:
    with fitz.open(pdf_path) as doc:
        return [p.get_text() for p in doc]


def renumber(nums: list[int]) -> list[int]:
    """Older papers restart numbering after GA (GA 1-10, CS 1-55). Map to one global 1..65 sequence."""
    out, offset, prev = [], 0, 0
    for n in nums:
        if n < prev and n <= 2:  # restart, not a stray out-of-order number
            offset += prev
        out.append(offset + n)
        prev = n
    return out


def parse_gate_paper(pages: list[str], ocr: bool = False) -> list[dict]:
    """Split GATE paper text into questions with options."""
    lines = []
    for ln in (ln.strip() for page in pages for ln in page.splitlines()):
        ln = re.sub(r"^Question Number\s*:\s*(\d{1,3})\b.*$", r"Q.\1", ln, flags=re.I)  # 2017-2021 online-exam layout
        ln = re.sub(r"^Q\.?\s*No\.?\s*(\d{1,2})\b", r"Q.\1", ln, flags=re.I)  # 2020: 'Q.NO. 9'
        if ocr:
            ln = re.sub(r"^Q\s*(\d{1,2})\b", r"Q.\1", ln)  # OCR drops the dot: 'Q9' (not for text layers: 'Q1' can be a table quarter)
        lines += re.split(r"\s+(?=\([B-D]\)\s)", ln)  # inline options '(A) x (B) y' -> one per line
    lines = [ln for ln in lines if ln and not NOISE.match(ln)]
    questions, cur = [], None
    for ln in lines:
        if m := Q_START.match(ln):
            if cur and cur["qno"] == m.group(1):  # same number again: a linked/common-data header, keep one question
                cur["_opt"] = None
                cur["stem"].append(m.group(3) or "")
                continue
            cur = {"qno": m.group(1), "stem": [m.group(3)] if m.group(3) else [], "options": {}, "_opt": None}
            questions.append(cur)
        elif cur is None:
            continue
        elif m := OPTION.match(ln):
            cur["_opt"] = m.group(1)
            cur["options"][m.group(1)] = m.group(2)
        elif cur["_opt"]:
            cur["options"][cur["_opt"]] = (cur["options"][cur["_opt"]] + " " + ln).strip()
        else:
            cur["stem"].append(ln)
    out = []
    for q in questions:
        text = re.sub(r"\s+", " ", " ".join(q["stem"])).strip()
        opts = [f"({k}) {re.sub(r'\s+', ' ', v).strip()}" for k, v in sorted(q["options"].items())]
        out.append({"qno": q["qno"], "text": text, "options": opts, "has_figure": int(bool(FIGURE.search(text)))})
    for q, g in zip(out, sequential(renumber([int(q["qno"]) for q in out]))):
        q["qno"] = str(g)
    return out


def sequential(nums: list[int]) -> list[int]:
    """GATE numbering is strictly increasing; repair OCR misreads (Q.28 read as 88) from neighbours."""
    out, prev = [], 0
    for n in nums:
        out.append(n if prev < n <= prev + 3 else prev + 1)
        prev = out[-1]
    return out


def is_clean(qs: list[dict]) -> bool:
    return [q["qno"] for q in qs] == [str(i) for i in range(1, 66)] and all(len(q["text"]) > 15 for q in qs)


TYPES = ("MCQ", "MSQ", "NAT")


def parse_gate_key(pages: list[str]) -> dict[str, dict]:
    """Official key PDF -> {qno: {qtype, section, answer, marks}}.
    Handles the three layouts seen 2017-2026: [qno, type, ...], [qno, session, type, ...], [qno, 'session type', ...]."""
    tok = [ln.strip() for page in pages for ln in page.splitlines() if ln.strip()]
    has_session = any("session" in t.lower() for t in tok[:15])
    rows = []
    for i, t in enumerate(tok):
        parts = t.split()
        if not parts or parts[-1] not in TYPES or len(parts) > 2 or i + 3 >= len(tok):
            continue
        if len(parts) == 1 and has_session and i >= 2 and tok[i - 2].isdigit():
            qno = tok[i - 2]
        else:
            qno = tok[i - 1]
        section, answer, marks = tok[i + 1], tok[i + 2], tok[i + 3]
        if qno.isdigit() and re.fullmatch(r"\d(\.\d)?", marks) and re.match(r"(GA|CS|DA)(\b|-?\d)", section):
            rows.append((int(qno), {"qtype": parts[-1], "section": section[:2], "answer": answer, "marks": float(marks)}))
    if not rows:  # OCR'd key: one table row per line, e.g. '12 6 MCQ CS B 1'
        for ln in (ln.strip() for page in pages for ln in page.splitlines()):
            if m := re.match(r"^(\d{1,2})\s+(?:\d{1,2}\s+)?(MCQ|MSQ|NAT)\s+(GA|CS|DA)\S*\s+(.+?)\s+(\d(?:\.\d)?)$", ln):
                rows.append((int(m.group(1)), {"qtype": m.group(2), "section": m.group(3), "answer": m.group(4), "marks": float(m.group(5))}))
    if not any(r["section"] in ("CS", "DA") for _, r in rows):
        return {}  # some mirrors file another subject's key under CS: no subject rows means wrong key
    return {str(g): r for g, (_, r) in zip(renumber([n for n, _ in rows]), rows)}


def gate_layout(qno: int, year: int) -> tuple[str, float]:
    """(section, marks) when no key. Up to 2013 GA came last (Q56-65); from 2014 it comes first."""
    if year <= 2013:
        if qno >= 56:
            return "GA", 1.0 if qno <= 60 else 2.0
        return "SUBJECT", 1.0 if qno <= 25 else 2.0
    if qno <= 10:
        return "GA", 1.0 if qno <= 5 else 2.0
    return "SUBJECT", 1.0 if qno <= 35 else 2.0


def ocr_pages(pdf_path) -> list[str]:
    """Local OCR (macOS Vision via ocrmac) for scanned papers. Gemini can't be used here: it refuses to
    transcribe widely published GATE papers verbatim (RECITATION). Returns one text per page with lines
    rebuilt top-to-bottom, so margin labels like 'Q.7' rejoin the text on the same row."""
    import io

    from ocrmac import ocrmac
    from PIL import Image

    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            img = Image.open(io.BytesIO(page.get_pixmap(dpi=200).tobytes("png")))
            boxes = ocrmac.OCR(img, recognition_level="accurate").recognize()  # (text, conf, (x, y, w, h)), y from bottom
            rows: list[list] = []
            for text, _, (x, y, w, h) in sorted(boxes, key=lambda b: -(b[2][1] + b[2][3] / 2)):
                yc = y + h / 2
                if rows and abs(rows[-1][0] - yc) < h * 0.5:
                    rows[-1][1].append((x, text))
                else:
                    rows.append([yc, [(x, text)]])
            lines = []
            for _, items in rows:
                items.sort()
                first = items[0][1]
                # 'Q.7' / 'Q9' alone on a row starts a question; put it on its own line for the parser
                if re.fullmatch(r"Q\.?\s*\d{1,2}", first) and len(items) > 1:
                    lines += [first.replace("Q", "Q.", 1) if not first.startswith("Q.") else first, " ".join(t for _, t in items[1:])]
                else:
                    lines.append(" ".join(t for _, t in items))
            pages.append("\n".join(lines))
    return pages


def ingest_gate(paper_pdf: Path, key_pdf: Path | None = None) -> dict:
    m = GATE_NAME.search(paper_pdf.name)
    if not m:
        raise ValueError(f"Name GATE files like GATE_2024_CS1.pdf / GATE_2024_DA.pdf, got {paper_pdf.name}")
    year, code, sample = int(m.group(1)), m.group(2).upper(), bool(m.group(3))
    qs = parse_gate_paper(page_texts(paper_pdf))
    if sample and [q["qno"] for q in qs] == [str(i) for i in range(1, 56)]:  # subject-only sample: slots 11-65
        for q in qs:
            q["qno"] = str(int(q["qno"]) + 10)
    elif not is_clean(qs):  # scanned, or a layout the text layer garbles: try local OCR
        try:
            oqs = parse_gate_paper(ocr_pages(paper_pdf), ocr=True)
            if is_clean(oqs) or abs(len(oqs) - 65) < abs(len(qs) - 65):
                qs = oqs
        except ImportError:
            pass  # ocrmac is macOS-only; keep the text-layer result
    if len(qs) < 40:
        raise ValueError(f"only {len(qs)} questions recovered")
    key = {}
    if key_pdf and key_pdf.exists():
        kt = page_texts(key_pdf)
        key = parse_gate_key(kt if sum(map(len, kt)) > 200 else ocr_pages(key_pdf))
    subject_section = "DA" if code == "DA" else "CS"
    for q in qs:
        n = int(q["qno"])
        k = key.get(q["qno"], {})
        sec, marks = gate_layout(n, year)
        q["section"] = k.get("section") or (sec if sec == "GA" else subject_section)
        if q["section"] not in ("GA", subject_section):
            q["section"] = subject_section
        q["marks"] = k.get("marks") or marks
        q["qtype"] = k.get("qtype") or ("MCQ" if q["options"] else "NAT")
        q["answer"] = k.get("answer")
    return {
        "exam": "GATE",
        "area": "gate_da" if code == "DA" else "gate_cs",
        "year": year,
        "title": f"GATE {year} {code}" + (" (sample)" if sample else ""),
        "questions": qs,
    }


# ---------- university papers (mid-sem / end-sem) ----------

UNI_Q = re.compile(r"^(?:Q\.?\s*(\d{1,2})[.):]?|(\d{1,2})[.)])(?:\s+(.*))?$", re.I)
UNI_EACH = re.compile(r"\(\s*(\d{1,2})\s*[x×]\s*\d{1,2}\s*\)")  # "(6 x 8)" = 6 marks each
UNI_SUB = re.compile(r"^\(?([a-j]|i{1,3}|iv|v|vi{0,3}|ix|x)\)\s*(.*)$")
UNI_MARKS = re.compile(r"[\[(]\s*(\d{1,2})\s*(?:marks?|m)?\s*[\])]\s*$", re.I)
UNI_NOISE = re.compile(r"(registration|roll no|total number of pages|time:|max\.? marks|q\.code|branch|answer (any|all|question)|figures in the right|^\d{3}-\d{2}/\d{2}/\d{4})", re.I)


def parse_university_paper(pages: list[str]) -> list[dict]:
    """Heuristic segmentation: top-level 'Q1/1.' and sub-parts 'a)/(i)'. Each sub-part is one row."""
    out, qno, section, each = [], None, None, None
    for raw in (ln.strip() for p in pages for ln in p.splitlines()):
        if not raw or UNI_NOISE.search(raw):
            continue
        if m := UNI_EACH.search(raw):
            each = float(m.group(1))
            raw = UNI_EACH.sub("", raw).strip()
            if not raw:
                continue
        if re.match(r"^part[\s-]*[ivx\d]+$", raw, re.I):
            section = raw
            continue
        if m := UNI_Q.match(raw):
            qno = m.group(1) or m.group(2)
            if m.group(3) and not re.match(r"^(answer|only|attempt)\b", m.group(3), re.I):
                out.append({"qno": qno, "text": m.group(3), "section": section, "marks": each})
            continue
        if m := UNI_SUB.match(raw):
            out.append({"qno": f"{qno or '1'}{m.group(1)}", "text": m.group(2), "section": section, "marks": each})
        elif out:
            out[-1]["text"] += " " + raw
    for q in out:
        if mm := UNI_MARKS.search(q["text"]):
            q["marks"] = float(mm.group(1))
            q["text"] = q["text"][: mm.start()].strip()
        q["text"] = re.sub(r"\s+", " ", q["text"]).strip()
        q["qtype"] = "DESCRIPTIVE"
    return [q for q in out if len(q["text"]) > 12]


# ---------- Gemini vision path (scanned pages, maths-heavy papers) ----------

class _VQ(BaseModel):
    qno: str
    text: str
    options: list[str]
    marks: float | None
    section: str | None
    has_figure: bool


class _VPage(BaseModel):
    questions: list[_VQ]


VISION_PROMPT = (
    "Transcribe every exam question on this page. Use LaTeX ($...$) for all mathematics. "
    "Keep question numbers exactly as printed (sub-parts as e.g. '2a'). Put MCQ options in `options` "
    "as '(A) ...'. Describe any figure briefly in square brackets inside the text and set has_figure. "
    "Skip headers, instructions and watermarks. Return an empty list if the page has no questions."
)


def vision_extract(pdf_path, gate: bool = False) -> list[dict]:
    if not llm.available():
        raise llm.LLMUnavailable(f"{Path(pdf_path).name} has no usable text layer; set GEMINI_API_KEY to OCR it with Gemini.")
    from concurrent.futures import ThreadPoolExecutor

    with fitz.open(pdf_path) as doc:
        pngs = [page.get_pixmap(dpi=150).tobytes("png") for page in doc]
    prompt = VISION_PROMPT + (" This is a GATE paper: qno is the printed integer only." if gate else "")
    groups = [pngs[i : i + 3] for i in range(0, len(pngs), 3)]  # 3 pages per request: fewer calls on the free tier

    def page(imgs):
        try:
            return llm.generate(prompt + " The images are consecutive pages.", _VPage, images=imgs, temperature=0, think="LOW", bulk=True).questions
        except Exception as e:  # one unreadable page should not sink the paper
            print(f"  page skipped: {e}")
            return []

    with ThreadPoolExecutor(6) as pool:
        pages = list(pool.map(page, groups))
    out = [q.model_dump() | {"has_figure": int(q.has_figure)} for page in pages for q in page]
    if gate:
        out = [q for q in out if re.fullmatch(r"\d{1,2}", q["qno"].strip())]
        for q, g in zip(out, renumber([int(q["qno"]) for q in out])):
            q["qno"] = str(g)
        merged = {}
        for q in out:  # a question split over two pages comes back twice: keep the fuller one
            if q["qno"] not in merged or len(q["text"]) > len(merged[q["qno"]]["text"]):
                merged[q["qno"]] = q
        out = sorted(merged.values(), key=lambda q: int(q["qno"]))
    return out


IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".heic"}


def _image_pdf(images: list[Path]) -> Path:
    """Photos of a paper -> one temporary PDF, so every later step sees the same input type."""
    import tempfile

    doc = fitz.open()
    for img in images:
        with fitz.open(img) as im:
            doc.insert_pdf(fitz.open("pdf", im.convert_to_pdf()))
    out = Path(tempfile.mkstemp(suffix=".pdf")[1])
    doc.save(out)
    return out


class _QPage(BaseModel):
    questions: list[_VQ]
    qtypes_seen: list[str]


STRUCTURE_PROMPT = (
    "Below is the raw text of an exam paper (from a PDF text layer or OCR, so it may be noisy). Split it into "
    "individual questions. Keep each question's wording; repair obvious OCR noise and write maths in LaTeX. "
    "qno as printed (sub-parts like '2a'). options as '(A) ...'. marks if stated. section if stated. "
    "Question formats may be new or unusual (match-the-following, assertion-reason, multi-select, numerical...): "
    "keep them intact and list every distinct format you saw in qtypes_seen.\n\n"
)


def ingest_any(path: Path | list[Path], area: str, exam: str, year: int | None, title: str | None = None) -> dict:
    """Ingest a new paper of any layout: digital PDF, scanned PDF, or photos of pages."""
    paths = path if isinstance(path, list) else [path]
    pdf = _image_pdf(paths) if paths[0].suffix.lower() in IMAGE_EXT else paths[0]
    pages = page_texts(pdf)
    source, ocr = "text layer", False
    if sum(map(len, pages)) < 300 * len(pages) or _looks_garbled("".join(pages)):
        try:
            pages, source = ocr_pages(pdf), "local OCR"
            ocr = True
        except ImportError:  # not macOS (e.g. the deployed server): let Gemini read the page images
            qs = vision_extract(pdf)
            return _finish(qs, area, exam, year, title or paths[0].stem, "gemini vision")

    qs = parse_gate_paper(pages, ocr=ocr)
    method = f"{source} + GATE parser"
    if not is_clean(qs):
        if llm.available():
            text = "\n".join(pages)
            chunks = [text[i : i + 8000] for i in range(0, len(text), 8000)]
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(4) as pool:
                parts = list(pool.map(lambda c: llm.generate(STRUCTURE_PROMPT + c, _QPage, temperature=0, think="LOW", bulk=True), chunks))
            qs = [q.model_dump() | {"has_figure": int(q.has_figure)} for p in parts for q in p.questions]
            method = f"{source} + Gemini structuring (formats: {', '.join(sorted({t for p in parts for t in p.qtypes_seen}))})"
        else:
            qs, method = parse_university_paper(pages), f"{source} + heuristic segmenter"
    return _finish(qs, area, exam, year, title or paths[0].stem, method)


def _finish(qs, area, exam, year, title, method) -> dict:
    for q in qs:
        if not q.get("qtype"):
            q["qtype"] = ("MCQ" if q.get("options") else "NAT") if exam == "GATE" else ("MCQ" if q.get("options") else "DESCRIPTIVE")
    if len(qs) < 3:
        raise ValueError(f"only {len(qs)} questions recovered ({method})")
    return {"exam": exam, "area": area, "year": year, "title": title, "questions": qs, "method": method}


def _looks_garbled(text: str) -> bool:
    """Maths PDFs often explode equations into one-symbol lines."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return bool(lines) and sum(len(ln) <= 2 for ln in lines) / len(lines) > 0.25
