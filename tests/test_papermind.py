"""One offline smoke test of the pipeline: parse -> label -> analyse -> plan -> assemble."""
from papermind import analysis, generate
from papermind.ingest import parse_gate_key, parse_gate_paper
from papermind.nlp import bloom

PAGE = """Computer Science and Information Technology (CS1)
Organising Institute: IIT Roorkee Page 1 of 64
General Aptitude
Q.1 – Q.5 Carry ONE mark Each
Q.1
Which ONE of the following statements is FALSE about a stack?
(A)
It is LIFO
(B)
Push adds an element
(C)
Pop removes the bottom element
(D)
It can be built from an array
Q.2
The number of edges in a tree with 10 vertices is ______ .
"""
KEY = "Q. No.\nSession\nQ. Type\nSection\nKey/Range\nMarks\n1\n1\nMCQ\nGA\nC\n1\n2\n1\nNAT\nCS-1\n9 to 9\n1\n"


def test_pipeline(monkeypatch):
    monkeypatch.setattr("papermind.config.GEMINI_API_KEY", "")  # offline: no network in tests
    qs = parse_gate_paper([PAGE])
    assert [q["qno"] for q in qs] == ["1", "2"] and len(qs[0]["options"]) == 4 and not qs[1]["options"]
    key = parse_gate_key([KEY])
    assert key["1"]["answer"] == "C" and key["2"] == {"qtype": "NAT", "section": "CS", "answer": "9 to 9", "marks": 1.0}
    assert bloom(qs[0]["text"]) == "Analyze"

    rows = [dict(id=i, paper_id=y, year=y, title=f"GATE {y} CS1", exam="GATE", qno=str(i), text=f"Consider a demand paging system {i}",
                 options=[], qtype="NAT", marks=2.0, section="CS", answer="1", unit="Operating System",
                 topic="Memory management and virtual memory: paging, page replacement", bloom="Apply", has_figure=0)
            for i, y in enumerate([2024, 2025, 2025, 2026], 1)]
    stats = analysis.analyze(rows)
    assert stats["units"][0]["unit"] == "Operating System" and stats["topics"][0]["hot_score"] == 100

    req = generate.parse_request_offline("mid-sem paper for OS")
    assert (req.exam, req.units) == ("MIDSEM", ["Operating System"])
    paper = generate.generate_paper(req, rows, seed=0)  # offline: filled from past questions
    assert paper["total_marks"] == 30 and paper["validation"]["questions"] == "4/10"  # shortfall is reported, not hidden
    assert not paper["validation"]["marks_ok"]
