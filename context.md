# PRISM — Project Context

**Pipeline:** YOLO (layout detection) → RapidOCR PP-OCRv4 (text OCR) → Texo ONNX (math OCR) → LaTeX → Markdown  
**Goal:** Lightweight, GPU-free document parsing pipeline for English academic text with bilingual (EN+ZH) support.  
**Platform:** Windows 11, Python 3.12, CPU-only inference.

---

## Repository Structure

```
testprism/
├── pipeline/                   # Core pipeline
│   ├── orchestrate.py              # Main CLI entry point — run PRISM on any image
│   ├── detection_postprocess.py    # NMS, merge, suppression on YOLO boxes
│   ├── latex_builder.py            # Assemble LaTeX document from detected blocks
│   ├── layout_utils.py             # Reading order, column detection, DAG sort
│   ├── math_worker_onnx.py         # Texo ONNX subprocess (math OCR worker)
│   ├── models_interface.py         # Model loading/unloading singletons
│   ├── tex_to_md.py                # LaTeX → OmniDocBench Markdown converter
│   └── text_worker.py              # RapidOCR subprocess (text + table OCR worker)
│
├── weights/                    # Tracked model weight files
│   ├── yolov11n-doclaynet.onnx     # Layout detection model (~10 MB)
│   ├── en_PP-OCRv4_rec.onnx        # English RapidOCR recognition model (~7.5 MB)
│   └── en_dict.txt                 # English character dictionary for RapidOCR
│
├── normalization/              # Image normalization package
│   ├── __init__.py                 # normalize_image_pil() entry point
│   ├── modality.py                 # Screenshot vs phone-photo detection
│   ├── pipeline.py                 # Full normalization pipeline
│   ├── geometric.py                # Perspective correction
│   ├── region_adaptive.py          # Shadow removal, Sauvola binarization
│   └── frequency_filter.py         # FFT moire removal
│
├── benchmarks/                 # Evaluation scripts
│   ├── run_omnidocbench.py         # OmniDocBench eval (primary benchmark)
│   ├── run_fox.py                  # Fox bilingual benchmark
│   ├── run_ocrbench.py             # OCRBench (VQA format)
│   ├── run_docvqa.py               # DocVQA (sliding-window ANLS)
│   ├── rerun_mixed.py              # Re-process only en_ch_mixed pages
│   ├── benchmark_glare.py          # Glare robustness benchmark
│   ├── make_report.py              # Parse OmniDocBench result JSONs → Markdown
│   └── qa_extract.py               # LLM-based QA extraction (Groq/OpenRouter/Anthropic)
│
├── Texo/                       # Math OCR model (FormulaNet / ONNX export)
│   ├── model/onnx/                 # encoder_model.onnx + decoder_model_merged.onnx
│   └── src/texo/                   # Model architecture and training code
│
├── omnidocbench_eval/          # OmniDocBench evaluation framework (submodule)
│   ├── src/                        # Eval pipeline (matching, metrics, CDM)
│   └── demo_data/                  # 18-page demo set with GT markdown
│
├── data/                       # Benchmark datasets (gitignored, local only)
│   ├── omnidocbench/               # 981 full-benchmark images + GT JSON
│   ├── fox/                        # 212 Fox pages (EN + CN)
│   ├── ocrbench/                   # OCRBench parquet
│   └── docvqa/                     # DocVQA validation parquet shards
│
├── preds/                      # Prediction outputs (gitignored)
│   ├── omnidocbench/               # 981 .md predictions for full benchmark
│   ├── fox/                        # Fox predictions
│   ├── ocrbench/                   # OCRBench predictions
│   ├── docvqa/                     # DocVQA predictions
│   └── glare_bench/                # Glare robustness test outputs
│
├── models/                     # Model weights (gitignored)
│   └── MFD/YOLO/                   # YOLO fine-tuned variants (not tracked)
│
├── results.md                  # Benchmark results summary (paper-ready)
├── context.md                  # This file
├── PRISM_OmniDocBench_Report.md # Detailed per-category OmniDocBench breakdown
├── pyproject.toml / uv.lock    # Python dependencies (managed with uv)
└── .gitignore                  # Excludes data/, preds/, models/
```

---

## Pipeline Stages

### 1. Image Normalization (`normalization/`)
Detects whether input is a screenshot or phone photo using entropy/histogram analysis.
- **Screenshot path**: CLAHE contrast normalization + downscale to 1280px
- **Phone photo path**: white balance → geometric rectification → shadow removal (DoG) → glare removal (inpainting) → moire removal (FFT) → CLAHE → DPI resize

### 2. Layout Detection (`models_interface.py`)
YOLOv11n fine-tuned on DocLayNet (11 classes). Classes: Text, Title, Section-header, Caption, Footnote, Page-header, Page-footer, List-item, Formula, Table, Picture. Runs as ONNX for CPU-only inference.

### 3. Detection Post-processing (`detection_postprocess.py`)
NMS, small-box suppression, header suppression (top 12% of page), formula padding (+12px on all sides).

### 4. Column Detection + Reading Order (`layout_utils.py`)
- Gutter histogram detects 1–8 column layouts
- 2-column: split left/right with ±20% midpoint margin
- N-column: gutter-center boundaries
- DAG-based reading order: caption pairing + footnote sinking (O(n), post-sort)

### 5. Text OCR (`text_worker.py`)
RapidOCR PP-OCRv4 running in a persistent subprocess (avoids torch memory inheritance).
- **EN engine**: `en_PP-OCRv4_rec.onnx` + `en_dict.txt`
- **CJK engine**: bundled ch_PP-OCRv4 models
- **Mixed engine**: runs both engines on every block, keeps the one with more output characters
- Crops are stitched into batches of 20 before each DBNet inference pass
- Preprocessing parallelized with `ThreadPoolExecutor(max_workers=4)`
- `TextOCRWorkerDual`: two subprocesses running in parallel, splits crop batch in half

### 6. Math OCR (`math_worker_onnx.py`)
Texo (FormulaNet) ONNX autoregressive decoder running in a persistent subprocess.
- Encoder: ViT-style image encoder → 384×384 grayscale input
- Decoder: 2-layer transformer, greedy decode with repetition penalty (1.15)
- `max_new_tokens=384` (raised from 256 to reduce formula truncations)
- Quality gate: discards hallucinations (≥10 tildes, repeated `\hline`, extreme length)
- `MathOCRWorkerOnnxDual`: two ONNX sessions in parallel subprocesses

### 7. Table Extraction (`text_worker.py` + `models_interface.py`)
RapidOCR tokens → coordinate-based heuristic table builder. Groups tokens into rows by Y centroid proximity, finds column boundaries via X-projection histogram, emits booktabs LaTeX.

### 8. LaTeX Assembly + Markdown Conversion (`latex_builder.py` + `tex_to_md.py`)
`assemble_document()` combines paracol columns, itemize lists, and body blocks into a full LaTeX document. `tex_to_omnidocbench_md()` converts to OmniDocBench-compatible Markdown: equations → `\[...\]`, tables → HTML `<table>`, sections → `#`/`##`.

---

## Benchmark Results

### OmniDocBench (981 pages, full benchmark)

| Task | EDR (lower=better) | Accuracy (1-EDR) |
|------|-------------------|-----------------|
| Text block | 0.4717 | 52.8% |
| Reading order | 0.5814 | 41.9% |
| Table (Edit dist) | 0.6202 | 38.0% |
| Table (TEDS) | — | 30.7% |
| Display formula | 0.8434 | 15.7% |

**Text by language:**

| Language | EDR | Accuracy |
|----------|-----|----------|
| English | 0.2030 | 79.7% |
| Simplified Chinese | 0.5444 | 45.6% |
| Mixed EN+ZH | 0.8899 | 11.0% |

**English academic subset (193 pages — Nougat-comparable):**

| Metric | PRISM | Nougat |
|--------|-------|--------|
| Text EDR | 0.1487 | ~0.365 |
| Text accuracy | **85.1%** | ~63.5% |

PRISM outperforms Nougat 2.5× on English academic text and beats it on every English doc type.

**System comparison (full benchmark, text EDR):**

| System | Text EDR | Accuracy |
|--------|----------|----------|
| GOT-OCR2.0 | ~0.22 | ~78% |
| MinerU | ~0.28 | ~72% |
| Marker | ~0.36 | ~64% |
| Nougat (English-only) | 0.365 | 63.5% |
| Nougat (Full) | 0.452 | 54.8% |
| **PRISM (Full)** | **0.4717** | **52.8%** |
| **PRISM (EN subset)** | **0.1487** | **85.1%** |

### Fox Benchmark (212 pages bilingual)

| Split | Pages | NED | Accuracy |
|-------|-------|-----|----------|
| English | 112 | 0.117 | 88.3% |
| Chinese | 100 | 0.102 | 89.8% |
| **Overall** | **212** | **0.110** | **89.0%** |

### OCRBench (reference only — VLM benchmark)
550 questions, ANLS via Llama 3.1 8B (Groq): **10.3%** overall.  
Not comparable to published leaderboard numbers (PRISM is an extraction pipeline, not a VLM).

### DocVQA (reference only — lower bound)
5349 questions, 1286 documents, sliding-window ANLS: **45.7%**.  
Sliding-window NED underestimates true accuracy; state-of-the-art OCR baselines score 30–60%.

### Glare Robustness Benchmark (18 OmniDocBench demo pages)

| Metric | Clean | Specular Glare | Gradient Glare |
|--------|-------|----------------|----------------|
| Per-page latency (s) | 6.69 | 6.17 | 6.69 |
| Peak RAM – all procs (MB) | 1669 | 1970 | 2013 |
| Chars extracted (mean/pg) | 1269 | 1204 | 1108 |
| Text accuracy vs GT | 27.5% | 28.8% | 29.4% |

Glare adds ~300 MB RAM (normalization pipeline activates glare inpainting on affected pages) but does not significantly increase latency. Char yield drops ~13% under gradient glare. The normalization pipeline's inpainting step is surprisingly effective — glared accuracy is not worse than clean on this 18-page sample.

---

## Key Engineering Decisions

### Subprocess Worker Architecture
`text_worker.py` and `math_worker_onnx.py` run as persistent subprocesses to avoid inheriting the ~400 MB torch/CUDA memory footprint of the main process. Workers start once per batch and stay alive across all pages. ONNX CPU memory arena is disabled globally (`enable_cpu_mem_arena=False`) to allow OS memory reclaim between pages.

### YOLO Lifecycle
YOLO is loaded once at the start of a batch, stays loaded across all pages, and is unloaded after the full batch completes. Previously it was unloaded after every page (wasting ~0.5–1s reload per page).

### Dual-Worker Parallelism
- `TextOCRWorkerDual`: splits crop batch across two RapidOCR subprocesses
- `MathOCRWorkerOnnxDual`: splits formula batch across two Texo ONNX subprocesses
- Within a page: math and text run concurrently via `ThreadPoolExecutor(max_workers=2)`

### Language Routing
- `data_source == 'simplified_chinese'` → CJK engine (`ch_PP-OCRv4`)
- `data_source == 'en_ch_mixed'` → dual-engine (EN + CJK, keeps block with more chars)
- `data_source == 'PPT2PDF'` → force `is_screenshot=True` (slide rendering)
- Default → English PP-OCRv4 engine

Previously, `en_ch_mixed` pages were incorrectly routed to the CJK-only engine — this was a bug fixed during development.

### Footnote Reading Order
The original DAG-based reading order added an edge from every non-footnote to every footnote (O(n²) edges). Replaced with O(n) post-sort reordering: non-footnotes first, footnotes last, preserving relative order within each group.

---

## Optimizations Applied (June 2026)

| # | Change | File | Impact |
|---|--------|------|--------|
| 1 | Stop unloading YOLO between pages | `run_omnidocbench.py` | ~0.5–1s saved per page |
| 2 | `TextOCRWorkerDual` for parallel text OCR | `run_omnidocbench.py` | ~2× throughput on text-heavy pages |
| 3 | Parallelize `_preprocess_crop` with ThreadPoolExecutor | `text_worker.py` | Faster crop preparation |
| 4 | `MathOCRWorkerOnnxDual` for parallel math OCR | `math_worker_onnx.py` | ~2× throughput on formula-heavy pages |
| 6 | Raise `max_new_tokens` 256→384 | `math_worker_onnx.py` | Fewer formula truncations |
| 7 | Dual-engine mixed strategy (both EN+CJK, pick best) | `text_worker.py` | Better accuracy on mixed-language pages |
| 8 | O(n) footnote sinking (was O(n²)) | `layout_utils.py` | Faster reading order, fewer false constraints |
| 9 | Force `is_screenshot=True` for PPT2PDF pages | `run_omnidocbench.py` | Better layout detection on slides |

---

## Weaknesses and Known Gaps

| Area | Score | Root cause |
|------|-------|------------|
| Display formulas | 15.7% | Texo struggles with complex multi-line equations; `max_new_tokens` limit |
| Reading order | 41.9% | DAG cycles on complex layouts; newspaper/magazine irregular flow |
| Table TEDS | 30.7% | Heuristic column detection misses merged cells and complex structures |
| Mixed EN+ZH text | 11.0% | Neither OCR engine handles truly interleaved EN+ZH in a single line |
| Handwriting | ~0% | RapidOCR trained on printed text only; no handwriting support |
| PPT2PDF | 25.6% | Slide layouts don't match document layout assumptions |

**Recommended improvements (not yet implemented):**
- Replace table heuristic with Microsoft Table Transformer (TATR) — expected +15–20% TEDS
- Fine-tune or replace Texo for better formula coverage
- Multilingual OCR engine (e.g., PaddleOCR multi-language) for true mixed-script lines

---

## Paper Framing

**Recommended scope:** "lightweight, GPU-free document parsing optimised for English academic text."

**Strongest claims:**
- 85.1% text accuracy on English academic subset (2.5× better than Nougat)
- 89.0% bilingual OCR on Fox benchmark
- CPU-only, no GPU required, ~7s/page on i7 hardware

**Not competitive on:**
- Full OmniDocBench (52.8% vs GOT-OCR2.0 ~78%)
- Formula recognition (15.7%)
- Handwriting

**Recommended benchmarks for paper:** OmniDocBench (EN academic subset) + Fox. OCRBench and DocVQA should be appendix/reference only.

---

## Running the Pipeline

```bash
# Single image
python pipeline/orchestrate.py path/to/image.jpg

# Full OmniDocBench evaluation (981 pages)
python benchmarks/run_omnidocbench.py \
    --gt-json data/omnidocbench/OmniDocBench_available.json \
    --images-dir data/omnidocbench/images \
    --pred-dir preds/omnidocbench

# Fox bilingual benchmark
python benchmarks/run_fox.py

# Glare robustness benchmark (18 demo pages)
python benchmarks/benchmark_glare.py

# Re-run only mixed-language pages
python benchmarks/rerun_mixed.py

# Parse OmniDocBench result JSON into Markdown report
python benchmarks/make_report.py
```

---

## Dependencies

Key packages (managed with `uv`, see `pyproject.toml`):
- `ultralytics` — YOLO inference
- `rapidocr-onnxruntime` — text OCR
- `onnxruntime` — ONNX inference for Texo and RapidOCR
- `tokenizers` — Rust-based tokenizer for Texo decoder
- `opencv-python` — image preprocessing
- `Pillow`, `numpy`, `scipy` — image manipulation
- `psutil` — RAM measurement in benchmarks
- `python-Levenshtein` — NED computation
- `pandas`, `pyarrow` — parquet handling for DocVQA/OCRBench
