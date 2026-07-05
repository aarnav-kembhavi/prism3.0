# PRISM — document image → structured LaTeX/Markdown, CPU-only

PRISM converts a document page image into structured LaTeX (+PDF) and
OmniDocBench-style Markdown: text, display math, tables (with spans),
pictures, captions, multi-column reading order — English and Chinese.

**OmniDocBench v1.6 (official, 1651 pages): Overall 78.46** — above Marker
(78.44) — with **~245 MB of weights, CPU-only, 4.7 s/page median, ≤2.2 GB
RAM**. Every system that scores higher runs 0.9B–241B-parameter VLMs or
multi-GB GPU pipelines.

| Doc | What's in it |
|---|---|
| [docs/context.md](docs/context.md) | Full architecture, every component, every decision + reasoning, ablations, config, repro commands |
| [docs/paperresults.md](docs/paperresults.md) | Results vs. all SOTA (v1.6 + v1.5 leaderboards, efficiency, model-size breakdown, controlled CPU head-to-head) |
| [paper.md](paper.md) | Experiment log — everything tried, measured, and rejected, with numbers |
| [docs/formula_fix_v2.md](docs/formula_fix_v2.md) | Deep dive: the formula-recall investigation (CDM 56.9 → 78.1) |
| [docs/pipeline_audit_2026-07-04.md](docs/pipeline_audit_2026-07-04.md) | Deep dive: tables, empty pages, routing fixes |

## Quick start

```bash
# single image → outputs/<stem>_output/{main.tex, main.pdf}
python pipeline/orchestrate.py path/to/page.png

# web UI
python app.py
```

Requires: Python 3.12, `onnxruntime`, `rapidocr-onnxruntime`, OpenCV, Pillow;
pdflatex/xelatex for PDF compilation. Model files under `models/`, `weights/`,
`Texo/model/onnx/` (see docs/paperresults.md §4 for the breakdown).
