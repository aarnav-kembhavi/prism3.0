# PRISM Benchmark Results

**Date:** 2026-06-27  
**Pipeline:** YOLO (yolov11n-doclaynet) → DocLayout YOLO (formula/table boost) → RapidOCR PP-OCRv4 (EN + CJK) → Texo ONNX (math) → TATR (table structure) → LaTeX → Markdown  
**Platform:** Windows 11, Python 3.12  
**Key changes since 2026-06-14:** DocLayout formula threshold 0.25→0.15 (+formula acc), DocLayout table class added (conf≥0.30), TATR table structure recognition (microsoft/table-transformer-structure-recognition-v1.1-all), 3-column layout detection (margin-trim + centroid-gap fallback)

---

## 1. OmniDocBench (Primary Benchmark)

**Dataset:** 981 pages (publicly available subset of 1,651 total)  
**Metric:** Edit distance ratio (EDR) — lower is better; Accuracy = 1 − EDR  
**Source:** [OmniDocBench](https://github.com/opendatalab/OmniDocBench)

### Top-line Scores

| Task | EDR | Accuracy | vs Jun-14 |
|------|-----|----------|-----------|
| Text block | 0.4449 | 55.5% | +2.7pp |
| Reading order | 0.5502 | 45.0% | +3.1pp |
| Table (Edit_dist) | 0.4763 | 52.4% | +14.4pp |
| Table (TEDS) | — | 43.6% | +12.9pp |
| Display formula | 0.7682 | 23.2% | +7.5pp |

> TEDS = Tree Edit Distance Similarity on HTML table structure (higher = better).  
> CDM metric unavailable on Windows (requires TeX Live + ImageMagick on Linux).

### Text Accuracy by Document Type

| Document Type | EDR | Accuracy | vs Jun-14 |
|---------------|-----|----------|-----------|
| academic_literature | 0.0785 | **92.2%** | +0.6pp |
| research_report | 0.1253 | **87.5%** | = |
| book | 0.1300 | **87.0%** | +1.6pp |
| magazine | 0.2400 | 76.0% | −0.1pp |
| colorful_textbook | 0.3547 | 64.5% | +0.3pp |
| newspaper | 0.4711 | **52.9%** | **+17.4pp** |
| exam_paper | 0.6713 | 32.9% | +0.9pp |
| PPT2PDF | 0.7437 | 25.6% | = |
| note | 0.9088 | 9.1% | +1.9pp |

### Text Accuracy by Language

| Language | EDR | Accuracy | vs Jun-14 |
|----------|-----|----------|-----------|
| English | 0.1486 | **85.1%** | +5.4pp |
| Simplified Chinese | 0.5326 | 46.7% | +1.1pp |
| Mixed EN+ZH | 0.8510 | 14.9% | +3.9pp |

### Text Accuracy by Layout

| Layout | EDR | Accuracy | vs Jun-14 |
|--------|-----|----------|-----------|
| Three column | 0.1479 | **85.2%** | **+32.6pp** |
| Multi-column (1+) | 0.2403 | 76.0% | +0.4pp |
| Double column | 0.3028 | 69.7% | = |
| Other | 0.5072 | 49.3% | +3.2pp |
| Single column | 0.5446 | 45.5% | +0.6pp |

> Three-column: +32.6pp from the centroid-gap + margin-trim column detection fix.  
> Newspaper text: +17.4pp from the same fix (newspaper pages are overwhelmingly 3-column).

### Reading Order by Language

| Language | EDR | Accuracy | vs Jun-14 |
|----------|-----|----------|-----------|
| English | 0.2891 | **71.1%** | +8.0pp |
| Simplified Chinese | 0.6418 | 35.8% | +0.9pp |
| Mixed EN+ZH | 0.7998 | 20.0% | +1.9pp |

### Reading Order by Layout

| Layout | EDR | Accuracy |
|--------|-----|----------|
| Three column | 0.2216 | **77.8%** |
| Double column | 0.3940 | 60.6% |
| Multi-column (1+) | 0.3946 | 60.5% |
| Other | 0.6059 | 39.4% |
| Single column | 0.6370 | 36.3% |

### Table TEDS by Document Type

| Document Type | TEDS | vs Jun-14 |
|---------------|------|-----------|
| newspaper | 59.6% | +4.8pp |
| book | 56.5% | +3.4pp |
| magazine | 52.8% | +10.5pp |
| exam_paper | 51.0% | **+36.8pp** |
| academic_literature | 43.8% | = |
| colorful_textbook | 41.7% | **+17.0pp** |
| PPT2PDF | 35.7% | +9.4pp |
| research_report | 32.1% | +5.1pp |
| note | 21.3% | +11.0pp |

### Nougat-Comparable English Subset

Filtered to: English only, excluding magazine / newspaper / note / PPT2PDF  
Pages: 193 text / 204 reading-order / 20 formula / 81 table

| Metric | PRISM EDR | PRISM Accuracy | vs Jun-14 |
|--------|-----------|----------------|-----------|
| Text block | **0.1238** | **87.6%** | +2.5pp |
| Reading order | **0.2804** | **72.0%** | +2.0pp |
| Display formula | 0.5972 | 40.3% | +8.1pp |
| Table (Edit_dist) | 0.4289 | 57.1% | +5.0pp |

### Per-Type Text vs Nougat (English Subset)

| Document Type | Pages | PRISM EDR | PRISM Acc | Nougat EDR | Nougat Acc |
|---------------|-------|-----------|-----------|------------|------------|
| academic_literature | 122 | **0.0785** | **92.2%** | 0.214 | 78.6% |
| book | 36 | **0.1469** | **85.3%** | 0.734 | 26.6% |
| colorful_textbook | 24 | **0.2411** | **75.9%** | 0.820 | 18.0% |
| exam_paper | 11 | **0.2943** | **70.6%** | 0.930 | 7.0% |

PRISM outperforms Nougat on every document type in the English subset.

### System Comparison (Full Benchmark, Text EDR)

| System | Text EDR | Notes |
|--------|----------|-------|
| GOT-OCR2.0 | ~0.22 | VLM, native CJK |
| MinerU | ~0.28 | Multi-model pipeline, CJK |
| Marker | ~0.36 | Layout + OCR pipeline |
| Nougat (English-only) | 0.365 | Dragged by non-academic English |
| Nougat (Full) | 0.452 | 0.998 EDR on Chinese pages |
| **PRISM (Full, Jun-14)** | **0.4717** | English pipeline, CJK added |
| **PRISM (Full, Jun-27)** | **0.4449** | +column detection, TATR, DocLayout |
| **PRISM (EN subset)** | **0.1238** | 2.9× better than Nougat EN-only |

---

## 2. Fox Benchmark (Bilingual OCR)

**Dataset:** 212 pages — 112 English + 100 Simplified Chinese  
**Metric:** 1 − NED (normalized edit distance); higher = better  
**Source:** Fox benchmark (focus_benchmark_test)

| Split | Pages | NED | Accuracy | vs Jun-14 |
|-------|-------|-----|----------|-----------|
| English | 112 | 0.119 | **88.1%** | −0.2pp |
| Chinese | 100 | 0.095 | **90.5%** | +0.7pp |
| **Overall** | **212** | **0.107** | **89.3%** | +0.3pp |

Chinese accuracy exceeds English, reflecting the strength of PP-OCRv4 CJK on clean printed text.  
Fox pages are single-column screenshots; layout improvements have minimal impact here.

---

## 3. OCRBench (For Reference Only)

**Dataset:** 550 questions across 4 task types (filtered from 1,000 total)  
**Metric:** ANLS via Llama 3.1 8B (Groq) reader  
**Note:** OCRBench is designed for end-to-end VLMs (image → answer directly). PRISM is an extraction pipeline, so these scores are not directly comparable to published OCRBench leaderboard numbers. Included for completeness only.

| Task Type | n | Score |
|-----------|---|-------|
| Key Information Extraction | 200 | 17.6% |
| Doc-oriented VQA | 200 | 10.7% |
| Handwriting Recognition | 50 | 0.0% |
| Handwritten Math Recognition | 100 | 0.0% |
| **Overall (4 types)** | **550** | **10.3%** |

> Results from Jun-14 run (v2 run in progress). Handwriting scores 0% by design — PRISM uses printed-text OCR only.

---

## 4. DocVQA (For Reference Only)

**Dataset:** 5,349 questions across 1,286 unique document images (validation set)  
**Metric:** ANLS via sliding-window fuzzy substring search (no LLM)  
**Note:** Standard DocVQA uses a reader model (BERT or LLM). Here we use direct fuzzy substring matching — tests whether the answer is recoverable from PRISM's output but may under-report for inference-requiring questions. Treat as a lower bound.

| Split | n | ANLS |
|-------|---|------|
| Validation (all) | 5,349 | **45.7%** |

> Results from Jun-14 run (v2 run in progress). State-of-the-art end-to-end models score 80–92% ANLS; OCR pipeline baselines score 30–60%.

---

## Summary

| Benchmark | Metric | Jun-14 | Jun-27 | Δ |
|-----------|--------|--------|--------|---|
| OmniDocBench (full) | Text accuracy | 52.8% | **55.5%** | +2.7pp |
| OmniDocBench (EN subset) | Text accuracy | 85.1% | **87.6%** | +2.5pp |
| OmniDocBench (full) | Table TEDS | 30.7% | **43.6%** | +12.9pp |
| OmniDocBench (full) | Table accuracy | 38.0% | **52.4%** | +14.4pp |
| OmniDocBench (full) | Formula accuracy | 15.7% | **23.2%** | +7.5pp |
| OmniDocBench (full) | Reading order | 41.9% | **45.0%** | +3.1pp |
| OmniDocBench (3-col) | Text accuracy | 52.6% | **85.2%** | **+32.6pp** |
| OmniDocBench (newspaper) | Text accuracy | 35.5% | **52.9%** | **+17.4pp** |
| Fox | OCR accuracy | 89.0% | **89.3%** | +0.3pp |
| OCRBench | ANLS (ref) | 10.3% | pending | — |
| DocVQA | ANLS (ref) | 45.7% | pending | — |

**Key improvements in this round:**
- **+32.6pp three-column text accuracy** via centroid-gap + margin-trim column detection (23/30 three-column pages now correctly detected vs 0 before)
- **+12.9pp table TEDS** via TATR (microsoft/table-transformer-structure-recognition-v1.1-all) + DocLayout table class
- **+7.5pp formula accuracy** via DocLayout formula threshold lowered to 0.15
- **+8.0pp English reading order** side-effect of correct three-column splitting

**Recommended benchmarks for paper:** OmniDocBench + Fox. These directly measure document parsing quality and bilingual OCR accuracy, which are PRISM's core claims.
