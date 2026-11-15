"""
Command line:
  python -m papermind.cli ingest-gate data/gate/raw        # GATE_<year>_<CS1|CS2|DA>.pdf (+ _key.pdf)
  python -m papermind.cli ingest-paper file.pdf --area mathematics_ii --exam ENDSEM --year 2025
  python -m papermind.cli train                            # distil labels into the runtime topic model
  python -m papermind.cli analyze gate_cs
  python -m papermind.cli generate "GATE DA mock focusing on machine learning" [--seed 1]
  python -m papermind.cli chat [--student name]           # personal study assistant
"""
import argparse
import json
from pathlib import Path

from papermind import db, generate, ingest, nlp
from papermind.analysis import analyze


def store(con, paper: dict, filename: str) -> None:
    nlp.label(paper["questions"], paper["area"], use_model=False)
    db.add_paper(con, paper["exam"], paper["area"], paper["year"], paper["title"], filename, paper["questions"])
    src = {q.get("label_source") for q in paper["questions"]}
    print(f"  {paper['title']}: {len(paper['questions'])} questions, labels via {', '.join(sorted(map(str, src)))}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="papermind")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("ingest-gate"); g.add_argument("folder", type=Path); g.add_argument("--force", action="store_true", help="re-ingest papers already in the DB")
    p = sub.add_parser("ingest-paper", help="any new paper: digital/scanned PDF or page photos (in order)")
    p.add_argument("files", type=Path, nargs="+"); p.add_argument("--area", required=True)
    p.add_argument("--exam", default="ENDSEM", choices=["GATE", "MIDSEM", "ENDSEM", "OTHER"]); p.add_argument("--year", type=int)
    p.add_argument("--title")
    sub.add_parser("train")
    a = sub.add_parser("analyze"); a.add_argument("area")
    c = sub.add_parser("chat", help="talk to the personal study assistant"); c.add_argument("--student", default="me")
    q = sub.add_parser("generate"); q.add_argument("request"); q.add_argument("--seed", type=int); q.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    con = db.connect()

    if args.cmd == "ingest-gate":
        done = {r[0] for r in con.execute("SELECT filename FROM papers")}
        for pdf in sorted(args.folder.glob("GATE_*.pdf")):
            if pdf.stem.endswith("_key") or (pdf.name in done and not args.force):
                continue
            try:
                store(con, ingest.ingest_gate(pdf, pdf.with_name(pdf.stem + "_key.pdf")), pdf.name)
            except Exception as e:
                print(f"  {pdf.name}: FAILED {e}")
    elif args.cmd == "ingest-paper":
        paper = ingest.ingest_any(args.files, args.area, args.exam, args.year, args.title)
        print(f"  read via {paper['method']}")
        store(con, paper, args.title or args.files[0].name)
    elif args.cmd == "train":
        for area in nlp.areas():
            rows = db.questions(con, area)
            if len({r["year"] for r in rows}) >= 2:
                print(json.dumps(nlp.train(rows, area)))
    elif args.cmd == "analyze":
        print(json.dumps(analyze(db.questions(con, args.area)), indent=1, default=list))
    elif args.cmd == "chat":
        from papermind import assistant

        print("PaperMind assistant — type 'exit' to quit, '/reset' to clear the conversation, '/progress' for stats.")
        while (msg := input("\nyou> ").strip()) not in ("exit", "quit"):
            if msg == "/reset":
                assistant.reset(con, args.student)
            elif msg == "/progress":
                print(json.dumps(assistant.progress(con, args.student), indent=1))
            elif msg:
                print("\npapermind>", assistant.chat(con, args.student, msg))
    elif args.cmd == "generate":
        req = generate.parse_request(args.request, nlp.areas())
        print("request ->", req.model_dump())
        paper = generate.generate_paper(req, db.questions(con, req.area), seed=args.seed)
        con.execute("INSERT INTO generated (request, paper) VALUES (?, ?)", (args.request, json.dumps(paper)))
        con.commit()
        text = json.dumps(paper, indent=1)
        (args.out.write_text(text) if args.out else print(text))
        print("validation ->", paper["validation"])


if __name__ == "__main__":
    main()
