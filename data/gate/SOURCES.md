# GATE corpus provenance

| Years | Papers | Source |
|---|---|---|
| 2025 | CS1, CS2, DA + keys | Official — IIT Roorkee, gate2025.iitr.ac.in/question-papers.html |
| 2026 | CS1, CS2, DA + keys | Official — IIT Guwahati, gate2026.iitg.ac.in/QPs-answer-keys.html |
| 2011–2024 | CS (one or two sets/year), DA 2024 + keys from 2017 | Mirror — www.gateexam.info (gateexaminfo.pages.dev) |
| 2024 (sample) | DA sample paper (subject section only) | Official — IISc, gate2024.iisc.ac.in |

**Mirror verification.** The mirror's 2025 DA paper is byte-identical to the official PDF and its 2025 CS1
paper + key are text-identical, so the mirror is treated as a faithful copy of official papers.

**Corrections applied**
- `GATE_2017_CS1_key.pdf` on the mirror is actually the EC (Electronics) key. The parser rejects any key
  without CS/DA rows, so 2017 CS1 is ingested without answers.
- `GATE_2013_CS.pdf` contained four booklet codes (A–D) of the same paper; only booklet A is kept.
- `GATE_2014_CS.pdf` contained three different session papers; split into `GATE_2014_CS1/2/3.pdf`.
  `GATE_2015_CS.pdf` (three sessions) and `GATE_2021_CS.pdf` (two sessions) were split the same way.
  Originals are in `data/gate/originals/`.
- The 2021 key is a scan whose OCR was unreliable and whose session could not be matched to a set, so 2021 is
  ingested without answers. The 2017 CS1 key on the mirror is the EC key (rejected).
- Scanned or irregular papers (2011, 2012, 2013, 2014, 2015, 2017, 2020, 2021, 2023) are read with local OCR
  (macOS Vision). Gemini vision cannot be used for these: it stops with `RECITATION` because the papers are
  widely published. Keys are unavailable before 2017.
