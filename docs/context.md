# PRISM — Full Project Context

## What It Is

PRISM (Pipeline for Robust Image-to-Structured Markup) is an end-to-end document image → LaTeX pipeline. It takes a single photo or screenshot of a document page (PNG/JPG) and produces a structured LaTeX source file (`main.tex`) and compiled PDF. It handles text, math formulas, tables, multi-column layouts, captions, headers, and footers.

**Target inputs:** scanned academic papers, textbooks, books, magazines, exam papers, resumes, newspapers — captured by phone camera or as digital screenshots.

**Supported languages:** English (primary), Chinese (CJK via xeCJK). Other languages exist in the benchmark dataset but are not a tested target.

**Platform:** Windows 11, Python 3.12.6, CPU-only inference (no CUDA required).

---

## Output

For each input image, PRISM writes to `outputs/<stem>_output/`:
- `main.tex` — structured LaTeX source
- `main.pdf` — compiled PDF (pdflatex for English, xelatex for Chinese)
- `crops/` — per-region image crops (for debugging)

---

## High-Level Pipeline (8 Stages)

```
Input image (PNG/JPG)
        │
        ▼
[NORMALISATION — Stage 1 + Stage 1.5]   ← normalization/
        │
        ▼
[LAYOUT DETECTION]                       ← pipeline/models_interface.py
  YOLOv11n-DocLayNet + DocLayout-YOLO
        │
        ▼
[DETECTION POSTPROCESS]                  ← pipeline/detection_postprocess.py
  NMS, overlap resolution, box refinement
        │
        ▼
[CONTENT EXTRACTION]                     ← pipeline/orchestrate.py
  Text OCR / Math OCR / Table structure
  (all run via persistent subprocess workers)
        │
        ▼
[READING ORDER & LAYOUT ASSEMBLY]        ← pipeline/layout_utils.py
  Column detection, caption pairing, footnote sinking
        │
        ▼
[LATEX GENERATION]                       ← pipeline/latex_builder.py
  Class-name → LaTeX environment mapping
        │
        ▼
[PDF COMPILATION]
  pdflatex (English) / xelatex (Chinese)
        │
        ▼
Output: main.tex + main.pdf
```

---

## Normalisation Pipeline

### Why Two Stages

Stage 1 normalises the **whole image** so YOLO can detect regions accurately.
Stage 1.5 normalises **individual crops** from the fidelity image (raw pixels, not the Stage 1 output) so OCR/TATR/Texo see clean content. The two stages serve different consumers and are not redundant — Stage 1's corrections never reach OCR; crops come from the pre-destructive fidelity image.

### Stage 1 — Whole-Image Normalisation (`normalization/`)

Execution order (deskew runs before modality detection):

| Step | Applies To | Operation |
|------|-----------|-----------|
| 0 | ALL | **Deskew** — projection profile method, rotate ±15°, pick angle with max derivative variance of row sums, apply affine warp if angle > 0.5° |
| 1 | ALL | **Capture Modality Detection** — 256-bin grayscale histogram, normalised Shannon entropy < 0.55 → SCREENSHOT, ≥ 0.55 → PHONE PHOTO |
| 2 | Photos | **White Balance** — gray world algorithm, equalise BGR channel means |
| 3 | Photos | **Geometric Rectification** — 3 strategies (morph gradient → Hough lines → Canny contour), perspective warp to frontal view |
|   |         | ← **FIDELITY IMAGE COPY taken here** (post-rectification, pre-destructive) |
| 4 | Photos | **Shadow Removal** — difference-of-Gaussians (divide by 51px Gaussian blur) |
| 5 | Photos | **Glare Inpainting** — LAB L > 230 mask, Telea inpainting (radius 5px), skip if glare < 0.1% |
| 6 | Photos | **Moiré Removal** — FFT notch filter per channel, 97th-percentile spike suppression, Gaussian mask (σ=3), IFFT |
| 7 | Photos | **Contrast Normalisation** — CLAHE on LAB L-channel (clip=2.0, grid=8×8) |
| 8 | Photos | **Smart DPI Resize** — target 250 DPI, cap shorter side at 1800px, never downscale below 0.5× |
| — | Screenshots | ← **FIDELITY IMAGE COPY taken here** (right after deskew) |
| 2s | Screenshots | **Contrast Normalisation** — CLAHE only |
| 3s | Screenshots | **Downscale if Oversized** — only if longest side > 1280px, no upscale |

**Stage 1 outputs:**
- Normalised image → passed to YOLO
- Fidelity image → used for all OCR crops (preserves original pixel values)
- `ModalityResult` (modality, entropy, confidence) → passed downstream

### Stage 1.5 — Per-Region Adaptive Preprocessing (`normalization/region_adaptive.py`)

Runs on each YOLO crop cut from the **fidelity image** after layout detection.

**Class-aware gating:**
- Picture → skip all corrections entirely
- Page-header / Page-footer → contrast normalisation only
- Formula → skip shadow removal (DoG gradients look like shadows on equations)
- All other classes → full pipeline below

**Per-region pipeline (in order):**

| Step | Applies To | Operation |
|------|-----------|-----------|
| 0 | ALL (adaptive threshold) | **Moiré Detection & Removal** — FFT on green channel, peak/mean ratio; phone 3.5×, screenshot 8.0×. Bypassed if Stage 1 already ran whole-image FFT (`skip_moire=True`) |
| 1 | Photos only | **Glare Detection & Inpainting** — L > 225, area > 2% of crop → Telea inpaint. Heavy glare (>15%) triggers step 2 |
| 2 | Photos only (triggered) | **Shadow Detection & Removal** — std-dev of ratio map ≥ 0.20 → DoG removal |
| 3 | ALL | **Contrast Normalisation** — RMS std-dev < 18.0 → CLAHE |

**Stage 1.5 output:** preprocessed crop + `RegionArtifactProfile` (moiré/glare/shadow/contrast: detected, fixed, severity score)

---

## Models

| Component | Model | Format | Size |
|-----------|-------|--------|------|
| Layout detection | YOLOv11n-DocLayNet | ONNX | 11 MB |
| Formula/table detection boost | DocLayout-YOLO (docstructbench, imgsz=1024) | ONNX | 72 MB |
| Text OCR detection | RapidOCR PP-OCRv4 det | ONNX (bundled in package) | ~4 MB |
| Text OCR recognition | RapidOCR PP-OCRv4 rec (English) | ONNX | 7 MB |
| Math OCR | Texo-distill encoder | ONNX | 52 MB |
| Math OCR | Texo-distill decoder (merged) | ONNX | 27 MB |
| Table structure | TATR v1.1-all INT8 | ONNX | 30 MB |
| **Total** | | | **~203 MB** |

**Notes:**
- TATR was originally 115 MB safetensors (PyTorch). Exported to ONNX FP32 (116 MB), then INT8 quantized to 30 MB (3.9× reduction). The main process never imports torch.
- Texo-distill is a distilled Donut-family encoder-decoder (77 MB safetensors FP32), trained for LaTeX formula recognition.
- Chinese OCR uses RapidOCR's built-in Chinese PP-OCRv4 engine (not a local file in weights/).

---

## Subprocess Worker Architecture

All heavy inference runs in persistent child processes to isolate memory and allow parallel dispatch. The main process never imports torch.

```
Main process (orchestrate.py)
├── TextOCRWorker (subprocess)        pipeline/text_worker.py
│     RapidOCR PP-OCRv4, ~130–160 MB
│     Tasks: text, text_chinese, text_mixed, probe, probe_chinese,
│            table, table_tokens (x1/x2/y1/y2 per token for TATR)
│
├── MathOCRWorkerOnnx (subprocess)    pipeline/math_worker_onnx.py
│     Texo ONNX encoder+decoder, ~200 MB
│     Tasks: math
│
└── TATROnnxWorker (subprocess)       pipeline/tatr_worker_onnx.py
      TATR INT8 ONNX, ~30 MB
      Tasks: detect → row/col grid → LaTeX tabular
```

Communication: `multiprocessing.Pipe`, spawn context, `(task, payload)` / `(status, result)` protocol.

**Worker startup sequence:** text+math workers start in background threads while YOLO runs, overlapping ~3s Texo load with normalisation+detection. Math and text extraction then run concurrently via `ThreadPoolExecutor`.

**RAM:** ~652 MB total (main + 3 workers) vs ~1,621 MB in the original in-process design.

---

## Layout Detection Detail

Two YOLO models run in sequence:

1. **YOLOv11n-DocLayNet** — primary layout detector. Outputs bounding boxes with 10 classes:
   `Text, Title, Section-header, Caption, List-item, Formula, Table, Picture, Page-header, Page-footer`

2. **DocLayout-YOLO** (docstructbench, imgsz=1024) — secondary model. Boosts formula and table detection confidence; rescues missed or low-confidence detections that the primary missed.

**Detection postprocessing** (`detection_postprocess.py`):
- Confidence threshold filter
- Class-aware NMS
- Overlap resolution (containment-based suppression)
- Box refinement
- Reading order sort

---

## Content Extraction

### Text OCR
- **Engine:** RapidOCR PP-OCRv4 (detection + recognition, both ONNX), runs in `TextOCRWorker` subprocess
- **Language routing:** sample up to 4 crops → count CJK codepoints vs ASCII chars → route to English-only / Chinese-only / mixed (both engines)
- **Preprocessing:** quiet-zone padding, max 1500px downscale; non-ASCII artifact filtering on output
- **Post-processing:** soft-hyphen cleanup, thousands-separator normalisation, citation bracket fixes

### Math OCR
- **Engine:** Texo-distill (custom distilled Donut-family FormulaNet encoder-decoder, ONNX), runs in `MathOCRWorkerOnnx` subprocess
- **Preprocessing:** Otsu binarisation + aspect-ratio padding for formula crops
- **Output:** LaTeX math strings; repetition-penalty and max_new_tokens guards against hallucination
- **Fallback:** placeholder inserted if output is degenerate

### Table Structure
- **Engine:** TATR (microsoft/table-transformer-structure-recognition-v1.1-all), INT8 ONNX, runs in `TATROnnxWorker` subprocess
- **Flow:**
  1. Table crop → `table_tokens` task to `TextOCRWorker` → returns `{text, x1, x2, y1, y2}` per token
  2. Tokens sent to `TATROnnxWorker` → DETR detection → rows and cols as xyxy boxes
  3. Each token assigned to best-overlap (row, col) cell
  4. Cell grid → LaTeX `tabular` with booktabs rules
- **Fallback:** coordinate heuristic if TATR returns no rows/cols

### Reading Order & Assembly (`layout_utils.py`)
- Semantic DAG: geometric order (top→bottom, left→right) as baseline
- Caption pairing: captions tied to nearest Picture or Table
- Footnote sinking: footnotes always follow body text
- Column detection: gutter analysis → split into 1/2/3-column zones → `paracol` environment for multi-column
- List-item grouping → `itemize` environments

---

## LaTeX Generation (`latex_builder.py`)

- Class → environment mapping:
  - Title, Section-header → `\section*{}`, `\subsection*{}`
  - Text → plain paragraph
  - Formula → `\[ ... \]` (display math)
  - Table → `\begin{tabular}{...}` with booktabs rules
  - Picture → `\includegraphics{crops/...}`
  - Caption → `\caption{}`
  - List-item → `\begin{itemize}`
- Preamble: `xeCJK` (Chinese detected) or `inputenc` + `fontenc` (English)
- Packages: `geometry`, `graphicx`, `booktabs`, `amsmath`, `amssymb`, `paracol`

---

## Web UI (`app.py`)

**Backend:** FastAPI, sequential job queue (one job processed at a time), background thread worker.

**Endpoints:**
- `POST /upload` — accept image, queue job, return `{job_id}`
- `GET /status/{id}` — `{status, message, queue_position}`
- `GET /pdf/{id}` — PDF bytes
- `GET /latex/{id}` — LaTeX source
- `GET /` — serve `index.html`

**Frontend:** split view — LaTeX source (syntax highlighted) left, PDF inline iframe right.

**Job lifecycle:** upload → queue → `orchestrate.py` subprocess → output written → PDF served → cleanup after 10 minutes.

**Current limitation:** each job spawns a fresh `orchestrate.py` subprocess so all models reload per upload (~10s overhead). Fixable by keeping orchestrate running as a long-lived daemon with persistent workers.

---

## Latency

**Pipeline throughput (warm — models loaded once, workers persistent across pages):**
- Mean: ~5.1 s/page, median ~4.7 s, range 4.9–5.3 s
- Measured: 26-page batch run (`run_bench_workers.py`)
- This is the true pipeline speed; the number to cite

**Web UI latency (cold — fresh subprocess per upload):**
- Mean: ~15.6 s, range 10–20 s
- Screenshots fastest (~10 s, skip normalisation Steps 1–6)
- Photos with many formulas/tables slowest (~20 s)
- Extra ~10 s is model reload overhead, not pipeline speed

**Model load breakdown (cold start):**
YOLO ~1 s, DocLayout-YOLO ~2 s, Texo ONNX ~1.5 s, TATR ONNX ~0.3 s, RapidOCR ~1 s

---

## Benchmark Results

**Dataset:** OmniDocBench, 981 pages
**Metric:** Edit Distance Rate (EDR, lower = better) / Accuracy (higher = better)

Four pipeline variants benchmarked:
- **v4** (2026-06-27): Full unconditional Stage 1 (baseline)
- **v5** (2026-07-01): Adaptive Stage 1 gated on detection scores — ABANDONED (table regression)
- **v6** (2026-07-01): Skip Stage 1 entirely (deskew + modality only) — **current best**
- **v7** (2026-07-01): Full Stage 1 + formula crops from pre-CLAHE fidelity image

### Text Block (v6 — current best)

| Document Type | v4 | v6 (skip-s1) | v7 (fid-formula) |
|---|---|---|---|
| academic_literature | 92.2% | 91.8% | 92.2% |
| research_report | 87.5% | 89.2% | 87.3% |
| book | 86.9% | 88.2% | 86.9% |
| magazine | 76.0% | 82.8% | 76.0% |
| colorful_textbook | 65.0% | 65.2% | 64.7% |
| exam_paper | 32.9% | 38.5% | 32.8% |

English: v4 85.1% → v6 **85.6%** → v7 85.1%

### Formula (Display Math, v6 — current best)

| Subset | v4 | v6 (skip-s1) | v7 (fid-formula) |
|---|---|---|---|
| English | 45.8% | **57.4%** | 46.4% |
| academic_literature | 63.1% | **70.2%** | 64.9% |

Key finding: the +11.6pp formula gain (v4→v6) comes from skipping ALL of Stage 1 — not just CLAHE. v7 (formula crops from pre-CLAHE fidelity image) only gains +0.6pp. DPI resize and white balance also affect Texo output, not just CLAHE on the crop.

### Table (v6 — marginal best)

Overall English: v6 EDR 0.410 (58.9% acc), TEDS 49.3%

| Document Type | TEDS v4 | TEDS v6 | TEDS v7 |
|---|---|---|---|
| academic_literature | 43.8% | 45.0% | 43.4% |
| newspaper | 59.6% | 57.4% | 51.1% |
| magazine | 52.8% | 37.3% | 40.2% |
| book | 56.5% | 35.6% | 36.6% |
| exam_paper | 51.0% | 33.0% | 30.6% |

TEDS ALL: v4 42.0%, v6 31.6%, v7 31.5%

Note: v6 and v7 table TEDS are nearly identical (~31.5% ALL). v4's 42% was from an earlier pipeline state and may reflect code differences unrelated to Stage 1. The table gap needs further investigation.

### Reading Order (v6 — best)

| Metric | v4 | v6 | v7 |
|---|---|---|---|
| ALL | 44.9% | **47.2%** | 44.2% |
| English | — | **71.0%** | 70.8% |
| academic_literature | 76.6% | 75.6% | 76.5% |
| magazine | 61.2% | 62.8% | 61.2% |

### Full Ablation Summary

| Metric | v4 | v5 | v6 | v7 |
|---|---|---|---|---|
| Text English | 85.1% | 84.4% | **85.6%** | 85.1% |
| Formula English | 45.8% | 47.1% | **57.4%** | 46.4% |
| Table TEDS ALL | 42.0%* | 30.9% | 31.6% | 31.5% |
| Table TEDS EN | 48.9%* | 47.6% | **49.3%** | 48.4% |
| Reading Order ALL | 44.9% | 44.2% | **47.2%** | 44.2% |

*v4 measured on earlier pipeline code, may not be directly comparable to v6/v7

**v6 (skip-stage1) wins on all metrics.** v5 (adaptive whole-image gating) and v7 (per-crop routing) both failed to improve over baseline. Current default: skip Stage 1 entirely.

---

## Comparison vs Nougat

English academic subset (academic_literature, book, colorful_textbook, exam_paper — v6 results):

| Document Type | PRISM v6 | Nougat | Delta |
|---|---|---|---|
| academic_literature | 91.8% | 78.6% | +13.2 pp |
| book | 88.2% | 26.6% | +61.6 pp |
| colorful_textbook | 65.2% | 18.0% | +47.2 pp |
| exam_paper | 38.5% | 7.0% | +31.5 pp |

PRISM v6 beats Nougat on every English document type. English subset: PRISM EDR 0.144 vs Nougat 0.365 — 2.5× better.

## Full-Benchmark Comparison (text EDR, lower = better)

| System | Scope | Text EDR |
|---|---|---|
| GOT-OCR 2.0 | Full (CJK+EN) | ~0.22 |
| MinerU | Full (CJK+EN) | ~0.28 |
| Marker | Full (CJK+EN) | ~0.36 |
| Nougat | Full | 0.452 |
| PRISM v6 | Full (EN only) | 0.410 |
| PRISM v6 | English subset | 0.144 |

PRISM v6's gap vs GOT-OCR/MinerU is driven by Chinese pages (~40% of dataset) and handwritten notes — not English document quality.

**Weak areas:** Chinese-only pages, mixed Chinese-English pages, handwritten notes, PPT-to-PDF slides. Table TEDS regression vs v4 baseline needs investigation.

---

## Codebase Structure

```
testprism/
├── app.py                          FastAPI web UI backend
├── pipeline/
│   ├── orchestrate.py              Main CLI entry point and job coordinator
│   ├── models_interface.py         In-process model wrappers (YOLO, RapidOCR, Texo)
│   ├── text_worker.py              RapidOCR subprocess worker
│   ├── math_worker_onnx.py         Texo ONNX subprocess worker
│   ├── tatr_worker_onnx.py         TATR INT8 ONNX subprocess worker (production)
│   ├── tatr_worker.py              (legacy) TATR PyTorch worker, not used by default
│   ├── detection_postprocess.py    YOLO output cleaning (NMS, overlap, box refine)
│   ├── layout_utils.py             Reading order, column detection, crop extraction
│   ├── latex_builder.py            LaTeX environment generation and document assembly
│   └── pix2tex_worker.py           (legacy) pix2tex subprocess, not used
├── normalization/
│   ├── pipeline.py                 Stage 1 orchestrator
│   ├── modality.py                 Histogram entropy → screenshot/photo classification
│   ├── geometric.py                Deskew + perspective rectification
│   ├── frequency_filter.py         White balance, shadow, glare, moiré, CLAHE
│   └── region_adaptive.py          Stage 1.5 per-crop preprocessing
├── Texo/                           Math OCR model (distilled Donut-family FormulaNet)
│   └── src/texo/model/formulanet.py
├── weights/
│   ├── yolov11n-doclaynet.onnx     Primary layout detector (11 MB)
│   └── en_PP-OCRv4_rec.onnx        English OCR recognition (7 MB)
├── models/
│   ├── doclayout_yolo_docstructbench_imgsz1024.onnx   Secondary detector (72 MB)
│   ├── tatr_structure.onnx         TATR FP32 ONNX (116 MB, source)
│   └── tatr_structure_int8.onnx    TATR INT8 ONNX (30 MB) ← used in production
├── omnidocbench_eval/              Evaluation harness (OmniDocBench fork)
├── scripts/
│   └── export_tatr_onnx.py         Export TATR PyTorch → ONNX FP32 → INT8
├── test_images/                    Test images (real/, synthetic/, rotation_benchmark/)
├── outputs/                        Per-image output folders (gitignored)
├── normalise.png                   Normalisation pipeline architecture diagram
├── arch.txt                        High-level pipeline architecture summary
├── arch2.txt                       Normalisation pipeline detailed spec
└── metrics.txt                     Model sizes, latency, benchmark results
```

---

## Key Dependencies

| Library | Role |
|---------|------|
| `fastapi` | Web UI backend |
| `ultralytics` | DocLayout-YOLO inference wrapper |
| `onnxruntime` | All ONNX inference: YOLO, RapidOCR, Texo, TATR (CPU) |
| `rapidocr-onnxruntime` | PP-OCRv4 text OCR (English + Chinese engines bundled) |
| `opencv-python` | Normalisation (CLAHE, FFT, inpainting, morphology, Hough, warp) |
| `Pillow` | Image I/O throughout the pipeline |
| `numpy` | Array operations throughout |
| `torch` + `transformers` | Texo/TATR training and ONNX export only; not needed at inference |
| `onnxruntime.quantization` | TATR INT8 quantization export |

---

## Requirements

- Python 3.12.6
- pdflatex (MiKTeX or TeX Live) in PATH for PDF compilation
- xelatex for Chinese documents
- CPU only — no GPU required
- ~650 MB RAM at inference (with all 3 workers running)
- Windows 11 (tested); Linux likely compatible with minor path adjustments
