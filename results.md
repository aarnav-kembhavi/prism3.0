# PRISM Benchmark Results

**Date:** 2026-06-14  
**Pipeline:** YOLO (yolov11n-doclaynet) → RapidOCR PP-OCRv4 (EN + CJK) → Texo ONNX (math) → LaTeX → Markdown  
**Platform:** Windows 11, Python 3.12

---

## 1. OmniDocBench (Primary Benchmark)

**Dataset:** 981 pages (publicly available subset of 1,651 total)  
**Metric:** Edit distance ratio (EDR) — lower is better; Accuracy = 1 − EDR  
**Source:** [OmniDocBench](https://github.com/opendatalab/OmniDocBench)

### Top-line Scores

| Task | EDR | Accuracy |
|------|-----|----------|
| Text block | 0.4717 | 52.8% |
| Reading order | 0.5814 | 41.9% |
| Table (Edit_dist) | 0.6202 | 38.0% |
| Table (TEDS) | — | 30.7% |
| Display formula | 0.8434 | 15.7% |

> TEDS = Tree Edit Distance Similarity on HTML table structure (higher = better).  
> CDM metric unavailable on Windows (requires TeX Live + ImageMagick on Linux).

### Text Accuracy by Document Type

| Document Type | EDR | Accuracy |
|---------------|-----|----------|
| academic_literature | 0.0838 | **91.6%** |
| research_report | 0.1252 | **87.5%** |
| book | 0.1457 | **85.4%** |
| magazine | 0.2389 | 76.1% |
| colorful_textbook | 0.3580 | 64.2% |
| newspaper | 0.6451 | 35.5% |
| exam_paper | 0.6805 | 32.0% |
| PPT2PDF | 0.7436 | 25.6% |
| note | 0.9284 | 7.2% |

### Text Accuracy by Language

| Language | EDR | Accuracy |
|----------|-----|----------|
| English | 0.2030 | **79.7%** |
| Simplified Chinese | 0.5444 | 45.6% |
| Mixed EN+ZH | 0.8899 | 11.0% |

### Text Accuracy by Layout

| Layout | EDR | Accuracy |
|--------|-----|----------|
| Multi-column (1+) | 0.2443 | 75.6% |
| Double column | 0.3031 | 69.7% |
| Three column | 0.4742 | 52.6% |
| Other | 0.5393 | 46.1% |
| Single column | 0.5514 | 44.9% |

### Reading Order by Language

| Language | EDR | Accuracy |
|----------|-----|----------|
| English | 0.3692 | 63.1% |
| Simplified Chinese | 0.6512 | 34.9% |
| Mixed EN+ZH | 0.8193 | 18.1% |

### Table TEDS by Document Type

| Document Type | TEDS |
|---------------|------|
| newspaper | 54.8% |
| book | 53.1% |
| academic_literature | 43.8% |
| magazine | 42.3% |
| research_report | 27.0% |
| PPT2PDF | 26.3% |
| colorful_textbook | 24.7% |
| exam_paper | 14.2% |
| note | 10.3% |

### Nougat-Comparable English Subset

Filtered to: English only, excluding magazine / newspaper / note / PPT2PDF  
Pages: 193 text / 204 reading-order / 20 formula / 81 table

| Metric | PRISM EDR | PRISM Accuracy |
|--------|-----------|----------------|
| Text block | **0.1487** | **85.1%** |
| Reading order | **0.2997** | **70.0%** |
| Display formula | 0.6784 | 32.2% |
| Table (Edit_dist) | 0.4793 | 52.1% |

### Per-Type Text vs Nougat

| Document Type | Pages | PRISM EDR | PRISM Acc | Nougat EDR | Nougat Acc |
|---------------|-------|-----------|-----------|------------|------------|
| academic_literature | 122 | **0.1044** | **89.6%** | 0.214 | 78.6% |
| book | 36 | **0.1878** | **81.2%** | 0.734 | 26.6% |
| colorful_textbook | 24 | **0.2397** | **76.0%** | 0.820 | 18.0% |
| exam_paper | 11 | **0.3137** | **68.6%** | 0.930 | 7.0% |

PRISM outperforms Nougat on every document type in the English subset.

### System Comparison (Full Benchmark, Text EDR)

| System | Text EDR | Notes |
|--------|----------|-------|
| GOT-OCR2.0 | ~0.22 | VLM, native CJK |
| MinerU | ~0.28 | Multi-model pipeline, CJK |
| Marker | ~0.36 | Layout + OCR pipeline |
| Nougat (English-only) | 0.365 | Dragged by non-academic English |
| Nougat (Full) | 0.452 | 0.998 EDR on Chinese pages |
| **PRISM (Full)** | **0.4717** | English pipeline, CJK added |
| **PRISM (EN subset)** | **0.1487** | 2.5× better than Nougat EN-only |

---

## 2. Fox Benchmark (Bilingual OCR)

**Dataset:** 212 pages — 112 English + 100 Simplified Chinese  
**Metric:** 1 − NED (normalized edit distance); higher = better  
**Source:** Fox benchmark (focus_benchmark_test)

| Split | Pages | NED | Accuracy |
|-------|-------|-----|----------|
| English | 112 | 0.117 | **88.3%** |
| Chinese | 100 | 0.102 | **89.8%** |
| **Overall** | **212** | **0.110** | **89.0%** |

Chinese accuracy exceeds English slightly, reflecting the strength of PP-OCRv4 CJK on clean printed Chinese text.

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

Handwriting tasks score 0% by design — PRISM uses RapidOCR trained on printed text only and makes no claim of handwriting support. The 10-17% on document tasks reflects the gap between extraction-then-QA versus native multimodal QA.

---

## 4. DocVQA (For Reference Only)

**Dataset:** 5,349 questions across 1,286 unique document images (validation set)  
**Metric:** ANLS via sliding-window fuzzy substring search (no LLM)  
**Note:** Standard DocVQA evaluation uses a reader model (BERT or LLM) to extract answer spans. Here we use direct fuzzy substring matching, which tests whether the answer string is recoverable from PRISM's output but may under-report accuracy for questions requiring inference. Treat as a lower bound.

| Split | n | ANLS |
|-------|---|------|
| Validation (all) | 5,349 | **45.7%** |

State-of-the-art end-to-end models (Donut, GPT-4V) score 80–92% ANLS. OCR pipeline baselines from the original DocVQA paper (2021) scored 30–60% using OCR + BERT reader. PRISM at 45.7% (lower bound) is consistent with that range.

---

## Summary

| Benchmark | Metric | Score | Notes |
|-----------|--------|-------|-------|
| OmniDocBench (full) | Text accuracy | 52.8% | All languages / doc types |
| OmniDocBench (EN subset) | Text accuracy | **85.1%** | Nougat-comparable English only |
| OmniDocBench (full) | Table TEDS | 30.7% | HTML structure similarity |
| Fox | OCR accuracy | **89.0%** | EN 88.3%, CN 89.8% |
| OCRBench | ANLS (ref only) | 10.3% | VLM benchmark, not apples-to-apples |
| DocVQA | ANLS (ref only) | 45.7% | Sliding window lower bound |

**Recommended benchmarks for paper:** OmniDocBench + Fox. These directly measure document parsing quality and bilingual OCR accuracy, which are PRISM's core claims. OCRBench and DocVQA use different evaluation paradigms (end-to-end VQA vs. extraction pipeline) and introduce methodological noise.
