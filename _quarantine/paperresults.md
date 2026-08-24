# PRISM — Paper Results: SOTA Comparison on OmniDocBench v1.5

**Overnight autonomous benchmark run — COMPLETE.** 11 systems self-run on the
same host; GPU-class giants cited from published numbers.

## Methodology

- **Benchmark:** OmniDocBench v1.5, **20-page stratified subset**
  (`compare20_subset.json`, mixed EN / zh / en-zh). VLMs are too slow for the
  full 981-page set on available hardware, so the subset is the common ground
  for all self-run competitors. (PRISM's own authoritative accuracy is the
  full 981-page run — see bottom.)
- **Metrics (OmniDocBench paper):** per-category **edit distance** (↓) for
  text / display-formula / table / reading-order; **TEDS** & **TEDS-structure**
  for tables (↑); **Overall** = mean of the 4 category edit distances. CDM
  (formula) omitted — needs TeX Live, unavailable on Windows.
- **Efficiency:** peak process-tree RSS (RAM), peak VRAM, per-page latency,
  weights on disk — measured in **isolated runs** on the same host.
- **Hardware:** Windows, 16-core CPU, **RTX 3070 Laptop (8 GB VRAM)**. CPU where
  feasible; GPU (4-bit for 7B models) otherwise. Efficiency compared CPU-vs-CPU;
  GPU models flagged (their GPU requirement is itself the differentiator).

---

## 1. Accuracy — 20-page subset (edit distance ↓, TEDS ↑)

Sorted by Overall. ⚑ = format/harness caveat (see notes) — read per-category.

| Model | Device | Text | Formula | Table | Order | TEDS | **Overall ↓** |
|---|---|---|---|---|---|---|---|
| PP-StructureV3 (server) | CPU | 0.050 | 0.256 | 0.117 | 0.185 | 0.853 | **0.152** |
| **PRISM (ours)** | **CPU** | **0.093** | 0.420 | 0.195 | 0.286 | 0.685 | **0.249** |
| MinerU 2 (pipeline) ⚑§ | CPU | 0.183 | 0.173 | 0.486 | 0.304 | 0.445 | **0.286** |
| Qwen2.5-VL-7B (4-bit) ⚑‡ | GPU | 0.191 | 0.206 | 0.547 | 0.344 | 0.440 | **0.322** |
| PRISM (old detectors) | CPU | 0.284 | 0.439 | 0.255 | 0.408 | 0.508 | 0.346 |
| GOT-OCR2 (580M) ⚑† | GPU | 0.196 | **0.168** | 1.000 | 0.282 | 0.000 | 0.412† |
| PP-Structure (mobile) | CPU | 0.166 | 0.711 | 0.674 | 0.314 | 0.234 | 0.466 |
| GraniteDocling-258M | CPU | 0.379 | 0.693 | 0.834 | 0.448 | 0.177 | 0.589 |
| Nougat-base ⚑¶ | GPU | 0.553 | 0.817 | 1.000 | 0.529 | 0.000 | 0.725 |
| SmolDocling-256M ⚑* | CPU | 0.599 | 1.000 | 0.867 | 0.528 | 0.234 | 0.748 |
| olmOCR-7B (4-bit) ⚑⌀ | GPU | (harness-limited — cite published) | | | | | |

### Format / harness caveats
- **† GOT-OCR2** emits **LaTeX** tables (OmniDocBench wants HTML) → table
  edit/TEDS are format-mismatched, inflating Overall. Its **text 0.196, formula
  0.168 (best formula recognizer measured), order 0.282** are fair reads.
- **‡ Qwen2.5-VL-7B** run 4-bit with vision tokens capped (~768×28×28 px) to fit
  8 GB — this *underestimates* it vs published full-res numbers.
- **§ MinerU** succeeded on 16/20 pages (4 hit a Windows batch cut-image bug →
  empty). Specialist pipeline (PP-DocLayout + UnimerNet), directly comparable.
- **¶ Nougat** is arXiv-English-only → collapses on this mixed (heavy-CJK) set +
  LaTeX-table format mismatch. Not representative on non-academic/CJK docs.
- **\* SmolDocling** DocTags→md doesn't align with OmniDocBench formula/table
  notation; text/order are fair.
- **⌀ olmOCR-7B** needs its official document-anchoring pipeline (PDF render +
  anchor text); bare-image self-run underperformed → **cite published accuracy**.
  Its efficiency (below) is a valid measurement.

---

## 2. Efficiency — measured, same host (isolated runs)

| Model | Device | Peak RAM | Peak VRAM | Latency median (s/pg) | Weights |
|---|---|---|---|---|---|
| **PRISM (ours)** | CPU | **1.3 GB** | — | **6.8** | ~240 MB |
| SmolDocling-256M | CPU | 2.3 GB | — | 92.2 | ~500 MB |
| MinerU 2 | CPU | 5.2 GB | — | ~11.3 | ~1.2 GB |
| PP-StructureV3 (server) | CPU | 8.2 GB | — | 58.2 | >1 GB |
| GOT-OCR2 (580M) | GPU | 2.7 GB | 2.7 GB | 16.8 | ~1.4 GB |
| Nougat-base | GPU | 3.3 GB | 1.7 GB | 16.5 | ~1.3 GB |
| Qwen2.5-VL-7B (4-bit) | GPU | 4.6 GB | 7.7 GB | 53.9 | ~5.5 GB |
| olmOCR-7B (4-bit) | GPU | 5.6 GB | 7.7 GB | 52.5 | ~5.5 GB |

---

## 3. Key findings

1. **PRISM is the efficiency leader by a wide margin** — 1.3 GB RAM, 6.8 s/page,
   CPU-only. Every VLM either needs a GPU or is far slower/heavier on CPU
   (SmolDocling 92 s/page; PP-StructureV3 8.2 GB / 58 s/page).

2. **Among CPU-deployable systems, PRISM is 2nd on accuracy (0.249)**, behind
   only PP-StructureV3-server (0.152) which costs **6× the RAM and 8.5× the
   latency**. PRISM beats MinerU (0.286), PP-Structure-mobile (0.466),
   GraniteDocling (0.589), SmolDocling (0.748).

3. **PRISM beats the GPU VLMs on this mixed benchmark** — Qwen2.5-VL-7B (0.322),
   GOT-OCR2 (0.412, table-inflated), Nougat (0.725) — while running on CPU at a
   fraction of the cost. (Caveat: the 7B VLMs are hardware-throttled here; their
   published full-precision numbers are higher.)

4. **Best formula recognizer measured: GOT-OCR2 (0.168)**, then MinerU (0.173),
   Qwen (0.206), PP-Structure-server (0.256). PRISM's formula (0.420) is its
   weakest category — consistent with the internal finding that formula
   recognition (not detection) is the remaining frontier.

5. **GPU 7B VLMs on an 8 GB laptop are impractical** for document parsing at
   scale: 52–54 s/page, near-max VRAM, and still not beating a 1.3 GB CPU
   pipeline on mixed real-world docs.

**Headline for the paper:** *PRISM occupies the efficient corner of the
accuracy/efficiency frontier — best-in-class footprint (1.3 GB, CPU-only, 6.8
s/page), 2nd-best accuracy among all CPU-deployable systems, and ahead of every
GPU VLM we could run on an 8 GB card, on mixed real-world documents.*

---

## 4. Cite from published (not fairly self-runnable on this host)

Accuracy for these should come from published OmniDocBench numbers; a fair local
run needs a GPU larger than 8 GB and/or the model's official harness:

- **dots.ocr (1.7B)** — `trust_remote_code` module-path bug on this stack
  (repo name contains a period); needs its own inference harness.
- **olmOCR-7B** — accuracy needs the official document-anchoring pipeline.
- **GPU-class / API SOTA**: GLM-OCR, MiniMax-M3, PaddleOCR-VL-1.5,
  InternVL2.5-8B+, Qwen2.5-VL-72B, HunyuanOCR, Qianfan-OCR, GPT-4o, Gemini,
  Mistral OCR, Claude.

---

## 5. PRISM full 981-page result (authoritative, all languages)

| Metric | PRISM (PP-DocLayout) |
|---|---|
| Text edit | 0.142 |
| Formula edit | 0.533 |
| Table edit / TEDS | 0.313 / 0.540 |
| Reading-order edit | 0.321 |
| **Overall edit** | **0.327** |

(The subset numbers above differ because the 20-page subset is CJK/formula-heavy
and stratified; the 981-page run is the headline accuracy figure. A CJK-table
OCR fix landed after this run and is expected to lift table TEDS toward ~0.68.)
