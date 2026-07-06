# PRISM — document image → structured LaTeX/Markdown, CPU-only

PRISM converts a document page image into structured LaTeX (+PDF) and
OmniDocBench-style Markdown: text, display math, tables (with spans),
pictures, captions, multi-column reading order — English and Chinese.

**OmniDocBench v1.6 (official, 1651 pages): Overall 86.35** — within 0.12 of
the MinerU GPU pipeline, above olmOCR-7B and Mistral OCR — with **~283 MB of
weights, CPU-only, 5.9 s/page median, ≤2.7 GB RAM**. On the v1.5 cut: **88.11,
the highest pipeline score**, above PP-StructureV3 and Gemini-2.5 Pro. Every
system that scores higher runs 0.9B–241B-parameter VLMs or multi-GB GPU
pipelines.

| Doc | What's in it |
|---|---|
| [docs/context.md](docs/context.md) | Full architecture, every component, every decision + reasoning, ablations, config, repro commands |
| [docs/paperresults.md](docs/paperresults.md) | Results vs. all SOTA (v1.6 + v1.5 leaderboards, efficiency, model-size breakdown, controlled CPU head-to-head) |
| [paper.md](paper.md) | Experiment log — everything tried, measured, and rejected, with numbers |
| [docs/formula_fix_v2.md](docs/formula_fix_v2.md) | Deep dive: the formula-recall investigation (CDM 56.9 → 78.1) |
| [docs/pipeline_audit_2026-07-04.md](docs/pipeline_audit_2026-07-04.md) | Deep dive: tables, empty pages, routing fixes |

## Repository layout

| Path | Contents |
|---|---|
| `pipeline/` | The parser: orchestration, layout (PP-DocLayoutV3), OCR workers, formula (Texo + geometry + render repair), tables (RapidTable/TATR), assembly, emission |
| `normalization/` | Capture-modality classifier + recognition-verified correction stack |
| `benchmarks/` | OmniDocBench / Fox runners and report tooling |
| `models/`, `weights/` | ONNX inference models (~283 MB total) |
| `omnidocbench_eval/` | Official eval harness (submodule) + `result/` score artifacts |
| `paper/` | WACV LaTeX sources; `paper.md` is the raw experiment log |
| `docs/` | Architecture, results vs. SOTA, per-section scores, deep dives |
| `app.py` + `web/` | FastAPI web UI (serves the exact benchmark build) |
| `.venv_rtable/`, `.venv_gpu/` | Purpose-built venvs: RapidTable child process; optional CUDA runtime. Kept at repo root because Windows venvs embed absolute paths (not relocatable) and `rtable_worker.py` addresses the first by path |

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
