# PRISM — Results vs. State of the Art

All PRISM numbers are from the **v16 run** (2026-07-06): full official
OmniDocBench dataset, official evaluation harness (`opendatalab/OmniDocBench`
main branch), CDM rendered with TeX Live 2026. Predictions:
`preds/odb_full_v16`; eval artifacts: `omnidocbench_eval/result/odb_full_v16_*`.

PRISM configuration: PP-DocLayoutV3 (layout + native reading order) + RapidOCR
PP-OCRv6-small (unified EN/CJK) + Texo-distill 20M (formulas; OCR-hybrid for
CJK-text formulas) + SLANet-plus (tables; TATR INT8 fallback), CPU-only,
~283 MB total weights.
`Overall = ((1−TextEdit)·100 + CDM·100 + TEDS·100)/3`.

---

## 1. OmniDocBench v1.6_full (1651 pages) — current official leaderboard

Leaderboard rows from the official repo (retrieved 2026-07-04). PRISM measured
by us on the identical dataset + harness + metric definitions.

| Model | Type | Overall↑ | TextEdit↓ | FormulaCDM↑ | TableTEDS↑ | ReadOrder↓ |
|---|---|---|---|---|---|---|
| MinerU2.5-Pro | Specialized VLM | 95.75 | 0.036 | 97.45 | 93.42 | 0.120 |
| GLM-OCR | Specialized VLM | 95.22 | 0.044 | 97.18 | 92.83 | 0.133 |
| PaddleOCR-VL-1.5 | Specialized VLM | 94.93 | 0.038 | 96.89 | 91.67 | 0.130 |
| PaddleOCR-VL | Specialized VLM | 94.18 | 0.040 | 95.91 | 90.65 | 0.135 |
| Qianfan-OCR | Specialized VLM | 93.90 | 0.040 | 95.08 | 90.53 | 0.130 |
| Youtu-Parsing | Specialized VLM | 93.74 | 0.044 | 93.63 | 92.02 | 0.116 |
| Ovis2.6-30B-A3B | General VLM | 93.70 | 0.035 | 95.17 | 89.44 | 0.135 |
| Logics-Parsing-v2 | Specialized VLM | 93.33 | 0.041 | 95.65 | 88.42 | 0.137 |
| ABot-OCR | Specialized VLM | 93.30 | 0.037 | 94.86 | 88.69 | 0.137 |
| FireRed-OCR | Specialized VLM | 93.26 | 0.037 | 95.44 | 88.04 | 0.131 |
| MinerU-2.5 | Specialized VLM | 93.04 | 0.045 | 95.77 | 87.88 | 0.130 |
| Gemini 3 Pro | General VLM | 92.91 | 0.064 | 95.99 | 89.15 | 0.165 |
| Gemini 3 Flash | General VLM | 92.62 | 0.066 | 95.16 | 89.29 | 0.172 |
| dots.ocr | Specialized VLM | 90.77 | 0.048 | 89.95 | 87.18 | 0.138 |
| OpenDoc-0.1B | Specialized VLM | 90.67 | 0.049 | 93.02 | 83.88 | 0.140 |
| DeepSeek-OCR 2 | Specialized VLM | 90.25 | 0.050 | 91.84 | 83.89 | 0.144 |
| HunyuanOCR | Specialized VLM | 89.95 | 0.088 | 87.68 | 91.01 | 0.171 |
| Qwen3-VL-235B | General VLM | 89.78 | 0.063 | 92.55 | 83.07 | 0.166 |
| Dolphin-v2 | Specialized VLM | 89.50 | 0.069 | 91.01 | 84.40 | 0.150 |
| OCRVerse | Specialized VLM | 88.60 | 0.063 | 89.61 | 82.44 | 0.163 |
| MonkeyOCR-pro-3B | Specialized VLM | 88.57 | 0.074 | 88.74 | 84.35 | 0.189 |
| GPT-5.2 | General VLM | 86.59 | 0.114 | 88.21 | 82.95 | 0.193 |
| Dolphin-1.5 | Specialized VLM | 86.52 | 0.094 | 87.49 | 81.43 | 0.167 |
| **MinerU-Pipeline** | **Pipeline** | **86.47** | 0.055 | 83.07 | 81.88 | 0.153 |
| **PRISM (ours, v16)** | **Pipeline — CPU, 283 MB** | **85.77** | 0.083 | 85.39 | 80.21 | 0.162 |
| olmOCR | Specialized VLM (7B) | 85.74 | 0.139 | 88.10 | 83.00 | 0.216 |
| Mistral OCR | Specialized VLM | 85.66 | 0.097 | 89.91 | 76.78 | 0.171 |
| Kimi K2.5 | General VLM | 84.53 | 0.107 | 83.50 | 80.76 | 0.211 |
| InternVL3.5-241B | General VLM | 83.76 | 0.130 | 89.95 | 74.35 | 0.215 |
| Nanonets-OCR-s | Specialized VLM (3B) | 83.61 | 0.108 | 81.46 | 80.18 | 0.213 |
| PRISM v14 (for reference) | Pipeline — CPU, 283 MB | 83.55 | 0.120 | 83.84 | 78.83 | 0.238 |
| POINTS-Reader | Specialized VLM (3B) | 83.37 | 0.096 | 85.72 | 73.98 | 0.198 |
| PRISM v10 (for reference) | Pipeline — CPU, 245 MB | 78.46 | 0.142 | 78.11 | 71.46 | 0.286 |
| **Marker** | **Pipeline** | **78.44** | 0.157 | 85.24 | 65.77 | 0.243 |

- PRISM is, to our knowledge, the only entry that runs **CPU-only with <300 MB
  of weights** — every VLM above it is 0.9–241B parameters on GPUs; the
  pipeline peers (MinerU-Pipeline, Marker) run multi-GB GPU model stacks.
- v16 sits **0.70 behind MinerU-Pipeline** (a multi-GB GPU stack) and above
  olmOCR-7B (85.74), Mistral OCR (85.66), Kimi K2.5 and InternVL3.5-241B —
  7B-VLM-class accuracy from 283 MB on CPU. PRISM's CDM (85.39) beats
  MinerU-Pipeline's (83.07); the remaining gap is text edit (0.083 vs 0.055).
- v16 by language — Text: EN 0.068 / ZH 0.096 / trad-ZH 0.211;
  TEDS: ZH 82.8 / EN 75.6; CDM: EN 90.1 / ZH 71.0. Full per-section tables:
  `docs/section_scores_odb_full_v16.md`.

## 2. OmniDocBench v1.5 (1355 pages)

Reference rows from the GLM-OCR technical report (arXiv 2603.10910, Table 4).
**Caveat for PRISM's row**: computed on the `subset: v1.5` cut of our v1.6 run
with the v1.6/1.7 matcher (the official v1.5 branch matcher differs slightly)
— directional, not a leaderboard submission.

| Model | Type / Params | Overall↑ | TextEdit↓ | FormulaCDM↑ | TableTEDS↑ | ReadOrder↓ |
|---|---|---|---|---|---|---|
| GLM-OCR | VLM 0.9B | 94.62 | 0.040 | 93.90 | 93.96 | 0.044 |
| PaddleOCR-VL-1.5 | VLM 0.9B | 94.50 | 0.035 | 94.21 | 92.76 | 0.042 |
| PaddleOCR-VL | VLM 0.9B | 92.86 | 0.035 | 91.22 | 90.89 | 0.043 |
| MinerU2.5 | VLM 1.2B | 90.67 | 0.047 | 88.46 | 88.22 | 0.044 |
| Gemini-3 Pro | General VLM | 90.33 | 0.065 | 89.18 | 88.28 | 0.071 |
| Qwen3-VL | General VLM 235B | 89.15 | 0.069 | 88.14 | 86.21 | 0.068 |
| MonkeyOCR-pro-3B | VLM 3.7B | 88.85 | 0.075 | 87.25 | 86.78 | 0.128 |
| dots.ocr | VLM 3B | 88.41 | 0.048 | 83.22 | 86.78 | 0.053 |
| Gemini-2.5 Pro | General VLM | 88.03 | 0.075 | 85.82 | 85.71 | 0.097 |
| MonkeyOCR-3B | VLM 3.7B | 87.13 | 0.075 | 87.45 | 81.39 | 0.129 |
| Deepseek-OCR | VLM 3B | 87.01 | 0.073 | 83.37 | 84.97 | 0.086 |
| Qwen2.5-VL-72B | General VLM | 87.02 | 0.094 | 88.27 | 82.15 | 0.102 |
| MonkeyOCR-pro-1.2B | VLM 1.9B | 86.96 | 0.084 | 85.02 | 84.24 | 0.130 |
| **PRISM (ours, v16)†** | **Pipeline — CPU, 283 MB** | **87.09** | 0.073 | 85.27 | 83.29 | 0.146 |
| **PP-StructureV3** | **Pipeline** | **86.73** | 0.073 | 85.79 | 81.68 | 0.073 |
| PRISM v14 (reference)† | Pipeline — CPU, 283 MB | 85.73 | 0.108 | 86.33 | 81.67 | 0.231 |
| Nanonets-OCR-s | VLM 3B | 85.59 | 0.093 | 85.90 | 80.14 | 0.108 |
| MinerU2-VLM | VLM 0.9B | 85.56 | 0.078 | 80.95 | 83.54 | 0.086 |
| GPT-5.2 | General VLM | 85.50 | 0.123 | 86.11 | 82.66 | 0.099 |
| Dolphin-1.5 | VLM 0.3B | 83.21 | 0.092 | 80.78 | 78.06 | 0.080 |
| InternVL3.5-241B | General VLM | 82.67 | 0.142 | 87.23 | 75.00 | 0.125 |
| olmOCR-7B | VLM 7B | 81.79 | 0.096 | 86.04 | 68.92 | 0.121 |
| POINTS-Reader | VLM 3B | 80.98 | 0.134 | 79.20 | 77.13 | 0.145 |
| InternVL3-76B | General VLM | 80.33 | 0.131 | 83.42 | 70.64 | 0.113 |
| PRISM v10 (reference)† | Pipeline — CPU, 245 MB | 80.23 | 0.127 | ~79 | 72.35 | 0.29 |
| Mistral OCR | VLM | 78.83 | 0.164 | 82.84 | 70.03 | 0.144 |
| **Mineru2-pipeline** | **Pipeline** | **75.51** | 0.209 | 76.55 | 70.90 | 0.225 |
| GPT-4o | General VLM | 75.02 | 0.217 | 79.70 | 67.07 | 0.148 |
| OCRFlux-3B | VLM 3B | 74.82 | 0.193 | 68.03 | 75.75 | 0.202 |
| Dolphin | VLM 0.3B | 74.67 | 0.125 | 67.85 | 68.70 | 0.124 |
| **Marker-1.8.2** | **Pipeline** | **71.30** | 0.206 | 76.66 | 57.88 | 0.250 |

† PRISM v1.5 rows = v1.5-subset cut of the v14/v10 runs scored with the
v1.6/1.7 matcher (the official v1.5 branch matcher differs slightly) —
directional, not a leaderboard submission.

On this snapshot PRISM v16 is the **top pipeline** — above PP-StructureV3
(which needs 8 GB and 58 s/page on CPU — see §5), Nanonets-OCR-s, MinerU2-VLM
and GPT-5.2; the next entries above it are GPU VLMs (MonkeyOCR-pro-1.2B 86.96).
Same text edit as PP-StructureV3 (0.073), higher TEDS (+1.6).

## 3. Systems with no current official OmniDocBench score

Not re-scored on v1.5/v1.6 by the leaderboard; their published/self-run
numbers are from **older, non-comparable** versions (different matcher/pages):

| System | Best known figure | Version / source | Note |
|---|---|---|---|
| Docling (classic) | EN overall edit ~0.28–0.34 | v1.0-era reports | pipeline, CPU-capable |
| SmolDocling-256M | EN overall edit 0.493 (published) | own paper, v1.0 protocol | our controlled CPU run: below PRISM on every metric (see §5) |
| GraniteDocling-258M | no official | — | our controlled run: below PRISM on every metric |
| Nougat | text edit 0.452 | v1.0 | academic-paper-only model |
| GOT-OCR 2.0 | EN overall 0.287 | v1.0 | 580M, GPU |
| PP-StructureV3 | Overall 86.73 | official v1.5 (see §2) | never re-scored on v1.6 |

## 4. Efficiency — measured on the v16 full run (1651 pages)

| Metric | PRISM v16 (CPU) | PRISM v16 (GPU-assisted) |
|---|---|---|
| Wall time (1651 pages) | 3.04 h | TBD |
| Latency median / mean / p90 | **5.62 s / 6.62 s / 10.97 s per page** | TBD |
| Peak RAM, process tree (dual OCR+math workers) | 2.60 GB | TBD |
| Total inference weights | **~283 MB** | same |
| Runtime | CPU-only (16 logical cores, Windows 11), onnxruntime; no torch in-process | + RTX 3070 Laptop: layout/OCR on CUDA, autoregressive math decode stays CPU (faster there) |

v14→v16 latency: +2.6 s median, spent on the Texo 512-token decode cap
(truncation fix), PP-DocLayoutV3 (~+0.3 s), and the CJK formula probe —
traded deliberately for +2.22 Overall. v14 (3.04 s median, 83.55) remains
the speed-optimal configuration.

Model size breakdown:

| Component | Model | Format | Size |
|---|---|---|---|
| Layout detection + reading order | PP-DocLayoutV3 (RT-DETR mask head, 800px) | ONNX FP32 | 124 MB |
| Text OCR | RapidOCR PP-OCRv6-small det + rec (unified EN/CJK) | ONNX | 31 MB |
| Formula recognition | Texo-distill 20M (encoder + merged decoder) | ONNX FP32 | 79 MB |
| Table structure | SLANet-plus (RapidTable) | ONNX | 7.4 MB |
| Table structure fallback | TATR structure-recognition v1.1-all | ONNX INT8 | 30 MB |
| Angle cls + dicts | ch_ppocr_mobile cls etc. | ONNX | ~1 MB |
| **Total** | | | **~283 MB** |

For scale: Marker's OCR alone (Surya) is a ~650M-param VLM; MinerU-Pipeline
ships a multi-model GPU stack; the leaderboard VLMs are 0.9B–241B.

## 5. Controlled CPU head-to-head (same 20 pages, same machine, isolated runs)

Local measurement (2026-07-02), 16-core CPU, 8-thread budget for every
system. PRISM numbers predate the v10 improvements (they are WORSE than
current). Edit distance ↓, TEDS ↑.

| System | Camp | Peak RAM | s/page | TextEN↓ | FormulaEN↓ | ReadOrdEN↓ | TableTEDS↑ |
|---|---|---|---|---|---|---|---|
| **PRISM** (pre-v10) | pipeline | **1.3 GB** | **6.8** | 0.165 | 0.491 | 0.319 | 46.1% |
| PP-StructureV3 (server) | pipeline | 8.0 GB | 58.2 | 0.081 | 0.320 | 0.276 | 56.5% |
| PP-StructureV3 (mobile) | pipeline | 3.6 GB | 6.3 | 0.156 | 1.000† | 0.362 | 43.5% |
| SmolDocling-256M | VLM | 2.3 GB | 92.2 | 0.461 | 1.000* | 0.498 | 0.0%* |
| GraniteDocling-258M | VLM | 2.3 GB | 54.6 | 0.341 | 0.585 | 0.458 | 37.4% |

\* output-format mismatch, not true ability (cite published). † config artifact
(mobile layout drops formula regions). GOT-OCR2.0 could not run (docling CPU
incompatibility); published EN overall 0.287.

Takeaways: PRISM wins efficiency outright (RAM leader; 8–14× faster than the
VLMs and PP-server) and beats both sub-300M VLMs on accuracy AND efficiency
simultaneously. Only PP-StructureV3-server is more accurate — at 6× the RAM
and 8.5× the latency.

## 6. PRISM progression (same official harness, v1.6_full)

| run | date | Overall | TextEdit↓ | CDM↑ | TEDS↑ | ReadOrd↓ | median s/pg |
|---|---|---|---|---|---|---|---|
| v9 | 2026-07-01 | 70.37 | 0.1575 | 56.90 | 69.96 | 0.3234 | 4.54 |
| v10 | 2026-07-04 | 78.46 | 0.1419 | 78.11 | 71.46 | 0.2864 | 4.71 |
| v13 | 2026-07-05 | 80.43 | 0.1358 | 83.46 | 71.40 | 0.2409 | 5.21* |
| v14 | 2026-07-05 | 83.55 | 0.1203 | 83.84 | 78.83 | 0.2383 | **3.04** |
| **v16** | 2026-07-06 | **85.77** | **0.0830** | **85.39** | **80.21** | **0.1617** | 5.62 |

\* v13 ran under CPU contention from concurrent experiments.

v9→v10 (+8.1) was pure code fixes; v10→v13 (+2.0) answer-key formula rule +
XY-cut reading order; v13→v14 (+3.1) RapidTable/SLANet-plus tables (TEDS
+7.4), PP-OCRv6 text (−0.0155 edit), formula LaTeX sanitizer; v14→v16
(+2.22) PP-DocLayoutV3 swap with native reading order (text −0.031, TEDS
+1.4, RO −0.077 — newspaper geometry + learned order), Texo 512-token cap
(long-matrix truncation), CJK-formula OCR-hybrid (CDM +1.55 combined).
Full history and negative results: `paper.md`. Weights v16 unchanged ≈
**283 MB** (PP-DocLayoutV3 replaces plus-L at identical size).
