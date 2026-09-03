# PRISM — Full Context for a New Session

> **Read this before touching anything.** It tells you what the system is, where
> every piece lives, what has been tried (with numbers), what is known NOT to
> work, and the operational rules that keep this machine alive. The experiment
> log with every measurement is `paper.md` (repo root); results vs. SOTA are in
> `docs/paperresults.md`; per-section score tables in
> `docs/section_scores_odb_full_v1{4,6,7,9}.md`.

---

## 1. What PRISM is and where it stands

PRISM converts a document page image into structured LaTeX + OmniDocBench-style
Markdown (text, display math, HTML tables with spans, figures, captions,
reading order; EN + ZH). Design constraints that define the project: **CPU-only,
~283 MB total weights, single-digit s/page, ≤~2.7 GB RAM**.

**Current confirmed results (v20 build, 2026-07-07, official harness):**

| Benchmark | Overall | text↓ | CDM↑ | TEDS↑ | RO↓ |
|---|---|---|---|---|---|
| OmniDocBench v1.6 full (1651 pg) | **86.62** | 0.0844 | 88.12 | 80.19 | 0.1644 |
| OmniDocBench v1.5 cut | **88.37** | 0.0767 | 89.51 | 83.27 | 0.1492 |

Position: v1.6 — **ABOVE MinerU-Pipeline (86.47, multi-GB GPU stack)** by 0.15,
above olmOCR-7B (85.74) and Mistral OCR (85.66); our CDM beats MinerU's by 5.05.
v1.5 — **top pipeline**, above PP-StructureV3 (86.73) and Gemini-2.5 Pro
(88.03). `Overall = ((1−text)·100 + CDM·100 + TEDS·100)/3`; reading order is
NOT part of Overall. **v20 = v19 + inline-math splicing (PRISM_INLINE_SPLICE);
see [[v20-inline-splice]] and paper.md.** The +0.25 over v19 is above the
±0.1 noise floor — a real cross, though 0.15 over MinerU is ~1.5× the noise, so
the robust claim is "+0.25 over our own v19 driven by CDM 87.56→88.12".

Perf (v20 CPU, 16-core laptop, dual-worker): median 5.91 s/pg, mean 7.16,
p90 12.71, p95 15.56, p99 19.99, p99.9 34.72, max 74.21; peak RAM 2.70 GB,
RAM p50 1877 / p95 2211 / p99 2342 MB. Weights 281.2 MB total / 251.3 active.
The speed-optimal config is v14 (83.55 @ 3.04 s median) — the last +3.1 points
deliberately cost +2.9 s (formula decode budget + bigger layout model).

---

## 2. Operational rules (violating these has burned us)

1. **SOLO execution.** The machine OOMs and has hard-crashed when two heavy
   jobs run concurrently (two benchmark runs, or a run + WSL CDM eval). One
   heavy job at a time, always. Tiny single-image inferences during a run are
   tolerable; a second worker-spawning run is not.
2. **Never point a subset eval at a full-run pred dir.** The eval's save_name
   comes from the pred dir name, so a 59-page "baseline" eval **silently
   overwrites the full run's result JSONs** (this destroyed odb_full_v15/v16
   metric artifacts once). Always `cp` preds into a distinctly named dir for
   baseline scoring (pattern: `preds/v18_base17`, `preds/odb_v19_v15cut`).
3. **Git Bash heredocs mangle backslashes.** Multi-line Python with `\\` inside
   `python - <<'EOF'` breaks silently (str.replace no-ops, SyntaxErrors on
   backslash literals). Write scripts to files (scratchpad) or use the Edit
   tool for anything containing backslashes.
4. **Background PowerShell task output is buffered.** A missing print in the
   log does NOT mean the code path didn't fire — verify by diffing outputs
   (`cmp` pred files across arms). Two "different" arms scoring byte-identical
   is how we caught read_order being silently dropped.
5. **Benchmark determinism:** `PRISM_NORM_STRICT=1` is set by the benchmark
   runner (keeps normalization byte-identical on benchmark pages).
   `PRISM_SKIP_EXISTING=1` resumes an interrupted run.
6. **A run leaves `_tmp_<page>/` LaTeX work dirs in its pred dir** (~20 MB
   each; 23k of them once accumulated to 36 GB). Sweep them after runs finish.
7. Timing budget: full 1651-page run ≈ 3–3.3 h; WSL v1.6 CDM eval ≈ 15 min;
   subset run (60–200 pg) ≈ 10–45 min; Windows subset eval (no CDM) ≈ 5–10 min.

### Eval commands
```powershell
# Full v1.6 with CDM (WSL; TeX Live 2026)
wsl -u root -e bash -lc 'export PATH=/usr/local/texlive/2026/bin/x86_64-linux:$PATH CDM_PDFLATEX=/usr/local/texlive/2026/bin/x86_64-linux/pdflatex CDM_SAVE_VIS=0 PYTHONPATH=/mnt/c/PROJECTS/s2l2/testprism/omnidocbench_eval; cd /mnt/c/PROJECTS/s2l2/testprism/omnidocbench_eval && python3 pdf_validation.py --config /mnt/c/PROJECTS/s2l2/testprism/data/omnidocbench_full/eval_cdm_v19full.yaml'

# Subset eval without CDM (Windows, fast)
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONPATH='C:\PROJECTS\s2l2\testprism\omnidocbench_eval'
cd omnidocbench_eval; python pdf_validation.py --config <yaml>
```
Config template: copy `data/omnidocbench_full/eval_cdm_v19full.yaml` (WSL
paths) or `eval_v3split.yaml` (Windows paths, no CDM). GT subsets already
built: `_split_pages.json` (211: newspapers+controls), `_fml_cjk_pages.json`
(73), `_inline_fml_pages.json` (59), `_v18_pages.json` (91), `_tblocr_pages.json`
(75), `_gpu_smoke.json` (10). Results land in `omnidocbench_eval/result/`.

---

## 3. Pipeline architecture (v19 = current defaults)

Flow per page: **normalize → layout detect → postprocess → route to
specialists → assemble in reading order → LaTeX → Markdown**.

### Entry points
- `pipeline/orchestrate.py <image>` — single-image CLI (what the web UI calls).
- `benchmarks/run_omnidocbench.py` — benchmark runner. Has its OWN detection
  wrapper `_layout_from_cache` (line ~97) — **any new detection dict key must
  be copied there AND in `orchestrate.run_detection`** (read_order and
  from_inline were both silently dropped there once).
- `app.py` — FastAPI web UI (port 8000), spawns orchestrate as subprocess.
- `benchmarks/run_fox.py` — Fox benchmark (212 pages, plain-text NED).

### Stage 1 — normalization (`normalization/`)
- `pipeline.py`: modality classifier (entropy/occupancy → screenshot vs
  phone_photo vs scan) routes to corrections. Screenshots/scans pass through
  (resolution cap only). Camera captures get: glare inpaint → shadow flatten →
  rectification → CLAHE, each **verification-gated**.
- `verified.py`: the probe — DBNet det @640px scores confident-text surface
  before/after; accept only ≥1.02×. This is the paper's normalization novelty.
- **`PRISM_NORM_STRICT=1` (benchmark) pins everything off for benchmark pages**
  — benchmark images are clean; the camera path is for real captures.
- Key measured finding (paper §verified): PP-OCRv6 is nearly invariant to
  photometric defects; open-loop correction taxes every page. Don't "improve"
  normalization expecting benchmark gains — it is score-neutral by design.

### Stage 2 — layout detection
- `pipeline/ppdoclayout_onnx.py`: raw-ORT wrapper. **PP-DocLayoutV3**
  (`models/ppdoclayoutv3/PP-DocLayoutV3.onnx`, 124 MB, default via
  `PRISM_PPDL_V3=1` in `models_interface.py`). 800×800, norm_type none (NOT
  ImageNet — per the official exported config.json). Output rows `[cls, score,
  x1,y1,x2,y2, read_order]`; a mask tensor is ignored. 25 classes; mapping
  `PPDL_V3_2PRISM`.
  - `read_order` is attached to each det → **model reading order**.
  - `inline_formula` class → mapped to `Formula` with `from_inline: True`
    (`PRISM_INLINE_FML_DISPLAY=1` default). This recovered ALL formulas on
    handwritten-notes/textbook pages (V3 labels them inline; dropping the
    class cost ~2 CDM in v16).
  - Legacy plus-L model still in `models/ppdoclayout/` (`PRISM_PPDL_V3=0`).
- `pipeline/detection_postprocess.py`: confidence gates (base 0.50, Formula
  0.30, Table 0.50 via `PRISM_PPDL_CONF/_TBL_CONF`), class-aware NMS,
  containment resolution, nearby-merge (merge inherits min read_order).

### Stage 3 — specialists (`pipeline/page_core.py` routes everything)
- **Text OCR**: `text_worker.py` — 2 persistent subprocess workers, RapidOCR
  1.4.4 with **PP-OCRv6-small det+rec swapped in** (`weights/PP-OCRv6_*_small
  .onnx`, `PRISM_OCR_V6=1`). Unified EN/CJK. `run_text_lines` = full-page
  det+rec used by rescue/probes. (rapidocr 3.x engine defaults are WORSE than
  v6-models-in-1.4.4 — do not "upgrade".)
- **Formulas**: `math_worker_onnx.py` — Texo 20M distill (79 MB,
  `Texo/model/onnx/`). Greedy decode, `_MAX_NEW_TOKENS` default **512**
  (`PRISM_FML_MAXTOK`; 256 truncated real matrices). `_sanitize` chain:
  KaTeX-ism rewrites → brace count balance → **`_render_repair`** (structural
  LaTeX repair: \left/\right paired per group+cell else demoted to \big;
  stray }/& fixes; array colspec widening; two-arg macro completion —
  validated 11/12 zero-CDM preds compile vs 4/12 before). `formula_v2.py` =
  ink-geometry layer (band split, fraction fusion, inline-FP guard with
  dense-grid exemption — **converted `from_inline` dets do NOT get the dense
  exemption**; that's the v19 text-guard fix).
  - **CJK hybrid** (`PRISM_FML_CJK=1`, page_core): on CJK pages each formula
    crop is probed with line OCR; ≥2 CJK chars → emit OCR-derived
    `\text{}`-wrapped LaTeX instead of Texo (Texo hallucinates on CJK).
- **Tables**: `page_core._extract_tables` — RapidTable **SLANet-plus** primary
  (7.4 MB) via stdio child in `.venv_rtable` (`rtable_worker.py` ↔
  `rtable_child.py`, length-prefixed PNG→JSON; protocol v2 can inject external
  OCR tokens — measured NO gain, off by default `PRISM_RTABLE_OCR_V6=0`).
  OCR coordinate-heuristic fallback when SLANet returns no cells.
- **Rescues** (page_core): uncovered-text rescue (`PRISM_TEXT_RESCUE=1`,
  full-page line OCR, orphan lines → synthetic Text dets); empty-page rescue
  (<30 chars → whole-page OCR).

### Stage 4 — reading order & assembly (page_core.build_document)
- **Model reading order** (`PRISM_RO_MODEL=1`): if ≥70% of dets carry
  `read_order`, sort by it (synthetics inherit nearest neighbour ±0.5) and
  skip ALL column logic. This halved RO (0.35→0.17 on the hard subset).
  **Counterintuitive measured fact: V3's finer boxes make geometric/XY-cut
  ordering WORSE (0.353→0.399) — never pair V3 with geometric ordering.**
- Geometric path (plus-L or `PRISM_RO_MODEL=0`): column split / DAG /
  recursive XY-cut — legacy fallback only.
- `PRISM_DROP_MARGINALIA=1`: headers/footers dropped (GT never scores them;
  emitting risks unmatched-pred penalties).

### Stage 5 — emission
- `latex_builder.py` (raw `<table` HTML passes through), `tex_to_md.py`
  (math AND raw HTML tables are quarantined from text rewrites — the %
  stripper once truncated tables at a literal %).

### GPU mode (optional, for the latency study only)
`PRISM_ORT_GPU=1` → CUDA EP in layout/OCR (via `onnx_config.ort_providers`
+ text_worker monkeypatch). Needs `venvs/gpu` python (onnxruntime-gpu==1.20.1
+ nvidia-*-cu12 wheels; the driver supports CUDA 12 only, NOT 13).
**Math worker stays on CPU** (`PRISM_ORT_GPU_MATH=0`): autoregressive decode is
SLOWER on GPU. `cudnn_conv_algo_search=HEURISTIC` is mandatory (EXHAUSTIVE
re-tunes per crop shape → 480 s pages). Layout: 0.78 s → 0.068 s. Accuracy
numbers are ALWAYS quoted CPU-only.

---

## 4. Version history (what worked, with numbers)

| ver | Overall v1.6 | What landed |
|---|---|---|
| v9 | 70.37 | baseline of record |
| v10 | 78.46 | formula ink-geometry (CDM 56.9→78.1), table spans, rescues |
| v13 | 80.43 | answer-key rule, XY-cut RO, marginalia drop |
| v14 | 83.55 | **RapidTable SLANet-plus (+7.4 TEDS), PP-OCRv6 swap (−0.016 text), formula sanitizer**; fastest build (3.04 s) |
| v15 | — | + uncovered-text rescue (subset-validated; superseded same day) |
| v16 | 85.77 | **PP-DocLayoutV3 swap (same 124 MB!) + model reading order** (text 0.120→0.083, TEDS +1.4, RO −0.077), fml maxtok 512, CJK hybrid |
| v17 | 86.35 | **inline_formula→Formula recovery** (CDM +2.17; V3 labels handwritten formulas inline; the dropped class silently zeroed notes/textbook pages) |
| v19 | 86.37 | inline dense-host guard fix (text −0.001) + LaTeX render repair (net 0 CDM — real fixes offset by chips the guard now drops) |
| v20 | **86.62** | **inline-math splicing (PRISM_INLINE_SPLICE): guard-dropped inline chips recognized by Texo, spliced into host text as $latex$ at char position; CROSSED MinerU 86.47. CDM 87.56→88.12 (free inline candidates rescue GT display formulas), text 0.0865→0.0844** |

(v18 = guard fix alone, killed at 25% and folded into v19.)

---

## 5. REJECTED with measurements — do not retry these

All logged in detail in `paper.md`:
- **Detector ensembling / dedicated MFD**: FP text-cost exceeds CDM gain.
- **Global confidence relaxation (0.15)**: fragments steal matches.
- **Table detection gate 0.30** (tried twice), **tiled detection** (partial
  recovery, never integrated; agate listings unaffected).
- **OCRv6-medium rec**: noise, +112 MB, 2× slower. Deleted from weights/.
- **rapidocr 3.9 engine defaults**: worse than v6-models-in-1.4.4 (ZH 0.165
  vs 0.086).
- **unitable**: torch dependency, low EV.
- **Clean-page enhancements** (upscale/sharpen/binarize): base wins every arm.
- **LaTeX canonicalization** (−0.0001), **native-res crops** (−0.003),
  **crop-prep variants** (noise), **moiré filter on paper** (erases text
  frequencies — screens only).
- **Emphasis markup** (bold/italic): breaks matcher pairing; plain text wins.
- **Feeding our v6 OCR tokens into SLANet** (`ocr_results` injection): TEDS
  72.98→72.94 = nothing; the child's internal OCR was not the bottleneck.
- **SLANeXt (PP-StructureV3's table model)**: 350 MB per variant — kills the
  budget. **PP-OCRv7 does not exist**; v6-small is current.
- **Colsplit/coalesce heuristics** (`PRISM_COLSPLIT/PRISM_COALESCE`, off):
  superseded by V3's column-accurate boxes; never validated a win.
- **WSJ dot-leader directory tables**: SLANet emits 1 column where GT wants 2;
  a split heuristic was considered and skipped (3 tables, high risk).

## 6. Remaining known weaknesses (and why we stopped)

From `docs/weakness_analysis_v15.md` + v19 section scores:
- **Newspaper tables TEDS ~0.62** (agate stock listings) — structure-model
  capacity; the model that fixes it is 350 MB.
- **Handwritten notes**: table TEDS 0.58, formula CDM ~0.72 — handwriting is
  a capacity wall for 20M-param recognizers.
- **ZH formulas CDM ~0.76 vs EN 0.90** — Texo has no CJK; the OCR-hybrid is
  the mitigation, a bigger recognizer is the fix.
- **Text edit 0.086 vs MinerU 0.055** — their OCR stack is far larger.
- **Inline math read as Unicode text** — needs true inline-math splicing
  (V3's inline_formula boxes + char-level positioning); biggest untapped
  lever but a multi-day change.
- historical_document (5 pages), trad-ZH (12 pages): tiny populations.

## 7. Repo map

```
pipeline/          the parser (see §3 for file-by-file)
normalization/     modality + verified corrections
benchmarks/        run_omnidocbench.py, run_fox.py, make_report.py
models/            ppdoclayoutv3/ (current), ppdoclayout/ (legacy plus-L)
weights/           OCR v6-small det+rec, en-v4 fallback, dicts
Texo/              formula model (model/onnx/) + its docs
omnidocbench_eval/ official harness + result/ (all score artifacts)
preds/             odb_full_v14..v19 mainline preds + perf.json each
data/              omnidocbench_full (GT + eval yamls + subset GTs), fox/
paper/             WACV LaTeX (compiles in WSL: pdflatex+bibtex)
paper.md           EXPERIMENT LOG — read before trying anything
docs/              context (this), paperresults, weakness analyses, section scores
venvs/gpu          CUDA runtime; .venv_rtable (root) = RapidTable child venv
                   (move to venvs/rtable when idle — worker checks both)
app.py + web/      FastAPI UI on :8000
```

## 8. Key env flags (defaults in code; benchmark = defaults + NORM_STRICT)

| Flag | Default | Meaning |
|---|---|---|
| PRISM_PPDL_V3 | 1 | PP-DocLayoutV3 layout (0 = legacy plus-L) |
| PRISM_RO_MODEL | 1 | detector reading order when available |
| PRISM_INLINE_FML_DISPLAY | 1 | inline_formula class → Formula path |
| PRISM_INLINE_SPLICE | 1 | guard-dropped inline chips → Texo → $latex$ spliced into host text (v20; +0.25 Overall) |
| PRISM_FML_CJK | 1 | OCR-hybrid for CJK-text formulas |
| PRISM_FML_MAXTOK | 512 | Texo decode cap |
| PRISM_TEXT_RESCUE | 1 | uncovered-text rescue |
| PRISM_OCR_V6 | 1 | PP-OCRv6-small models in RapidOCR 1.4.4 |
| PRISM_RTABLE | 1 | SLANet-plus primary table recognizer |
| PRISM_RTABLE_OCR_V6 | 0 | inject our OCR into SLANet (measured no-op) |
| PRISM_NORM_STRICT | benchmark sets 1 | pin normalization off for benchmark |
| PRISM_NORM_VERIFY | 1 | probe-gated camera corrections |
| PRISM_ORT_GPU / _MATH | 0 / 0 | CUDA EP (latency study only) |
| PRISM_SKIP_EXISTING | 0 | resume interrupted benchmark run |
| PRISM_COLSPLIT / PRISM_COALESCE | 0 | legacy heuristics, superseded by V3 |

## 9. Do NOT

- Do not flip any §8 default without a subset A/B — each one is measured.
- Do not run two heavy jobs at once (see §2.1).
- Do not score a subset against a full-run pred dir (see §2.2).
- Do not chase <0.1 Overall with full runs — that's the noise floor.
- Do not swap rapidocr versions or "update" onnxruntime in the main env.
- Do not touch `.venv_rtable` while any benchmark run is alive.
- Do not trust absence of a log line in a buffered background task — diff
  the outputs.
- Do not delete `preds/odb_full_v19` or `omnidocbench_eval/result/*v19*` —
  final paper numbers live there. `perf.json` per pred dir = latency record.

## 10. In flight / next steps (as of 2026-07-07 ~02:30)

- Fox benchmark run of v19 → `preds/fox_v19` (fills paper §fox `\todo`).
- GPU latency full run on v19 (venvs/gpu python + `PRISM_ORT_GPU=1`; V3 is
  default now) → fills paperresults §4 GPU column + paper efficiency prose.
- RAM percentiles: the runner now records a RAM time-series in perf.json
  (added post-v19) — any future run gets RAM p50/p90/p95/p99 automatically.
- Move `.venv_rtable` → `venvs/rtable` once no run is alive.
- Paper: title/abstract/experiments current at v19 numbers; sweep remaining
  `\todo`s (Fox, GPU), final compile (WSL pdflatex+bibtex), push.
