# STATUS — WACV results-hardening (Phase 0 inventory)

Branch: `wacv-results-hardening`. Author: automated pass against `instructions.txt`.
Date: 2026-07-07. Nothing here modifies a metric or scoring harness; every number
cited is traceable to a logged run under `omnidocbench_eval/result/`.

---

## 0.1 Harness invocation, harness version, TeX Live

- **v1.6 accuracy (with CDM)** is produced by the vendored OmniDocBench eval
  (`omnidocbench_eval/pdf_validation.py`) under **WSL**, TeX Live 2026:
  ```
  wsl -u root -e bash -lc 'export PATH=/usr/local/texlive/2026/bin/x86_64-linux:$PATH \
    CDM_PDFLATEX=/usr/local/texlive/2026/bin/x86_64-linux/pdflatex CDM_SAVE_VIS=0 \
    PYTHONPATH=.../omnidocbench_eval; cd .../omnidocbench_eval && \
    python3 pdf_validation.py --config .../eval_cdm_v19full.yaml'
  ```
  Matcher: `quick_match`. Composite: `Overall = ((1-TextEdit)*100 + CDM + TEDS)/3`
  where TEDS is the **page-averaged** `table.page.TEDS.ALL` (NOT micro-averaged).
- **TeX Live**: 2026 — `pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026)` (WSL).
- **Harness version/provenance**: vendored copy under `omnidocbench_eval/`; each run
  writes `*_runtime_environment.json/.log` capturing package versions — use those
  for exact provenance in the paper.

## 0.2 Per-page component scores (required for bootstrap CIs) — AVAILABLE

The harness already emits per-page/per-instance JSONs; **no metric change needed**.
For the final PRISM build (`odb_full_v20`) all of these exist in
`omnidocbench_eval/result/`:

| Component | File | Structure |
|---|---|---|
| Text edit | `..._text_block_per_page_edit.json` | `{page.png: edit}` (1557 pages) |
| Reading order | `..._reading_order_per_page_edit.json` | `{page.png: edit}` (1638 pages) |
| Formula edit | `..._display_formula_per_page_edit.json` | `{page.png: edit}` (313 pages) |
| Formula CDM | `..._display_formula_per_sample_CDM.json` | `{page.png_[i]: CDM}` (2352 instances) |
| Table edit | `..._table_per_page_edit.json` | `{page.png: edit}` (458 pages) |
| Table TEDS | `..._table_per_table_TEDS.json` | `{page.png_[i]: {TEDS,...}}` (665 tables) |

**Design note for the composite bootstrap:** text/order are per-page; CDM is
per-formula-instance; TEDS is per-table-instance. A per-page composite must define
how pages with no formula/table contribute (the official aggregate omits them from
that component). Plan: resample **page indices**; per resample recompute text mean,
CDM mean over the formula instances on the resampled pages, TEDS page-average over
the resampled pages — matching the official aggregation, applied to the resample.
This keeps the estimator identical to the point estimate. (Script = Phase 1a.)

## 0.3 Config system, v20 assembly, leave-one-out feasibility

The pipeline is assembled entirely from **env flags** (defaults live in code; the
benchmark runner adds `PRISM_NORM_STRICT=1`). Leave-one-out is a flag flip each.
Verified flag names (grepped from source):

| Ablation variant (instructions 1c) | Flag to disable | Note |
|---|---|---|
| formula ink-geometry off | `PRISM_FML_V2=0` | falls back to raw Texo on layout crops |
| native reading order off | `PRISM_RO_MODEL=0` | falls back to geometric/XY-cut (⚠ V3 boxes make geometric WORSE — expected regression) |
| verified normalization off | `PRISM_NORM_VERIFY=0` | **but** benchmark sets `PRISM_NORM_STRICT=1` which pins normalization OFF already; the verified/open/none study (1e/2f) requires `PRISM_NORM_STRICT=0` to engage the path at all |
| inline-math splicing off | `PRISM_INLINE_SPLICE=0` | = v19 behaviour |
| class-aware detection gates off | uniform `PRISM_PPDL_CONF` (drop `PRISM_PPDL_TBL_CONF` split) | removes per-class thresholds |

Table 9 (cumulative stages) is reproduced by the version sequence v9→v20; each row
already has a logged `..._metric_result.json`.

## 0.4 Baselines: installed / runnable / missing

Not installed as standalone envs, **but** `benchmarks/compare/` has runners and
pre-existing predictions from an earlier comparison pass:

- **Runners present**: `run_mineru.py`, `run_ppstructure.py`, `run_smoldocling.py`,
  `run_gpu_vlm.py`, `run_docling_vlm.py`, `run_texo_formulas.py`.
- **Predictions present**: `preds_mineru` (**22-page subset only**), `preds_got2`,
  `preds_dots`, `preds_qwen25vl`, `preds_olmocr`, `preds_nougat`, ppstructure
  (`cmp_ppstructure*` results). All scored under the same harness (per-page JSONs
  exist, e.g. `preds_mineru_quick_match_*_per_page_edit.json`).
- **MISSING for the crux statistic**: MinerU (and Marker) predictions on the
  **full identical 1651-page set**. The paper's MinerU `86.47` is the **published
  leaderboard** number, not a local same-harness run. The paired bootstrap
  (PRISM−MinerU on identical pages/harness/TeXLive) therefore needs a MinerU
  full-1651 run. → **BLOCKED on GPU inference (Vast.ai or local RTX 3070).**
- PP-StructureV3: was run (server+mobile) on the 20-page CPU head-to-head; full
  1651 not present.

## 0.5 Datasets present locally

| Dataset | Path | Notes |
|---|---|---|
| OmniDocBench v1.6 full (1651) | `data/omnidocbench_full` | GT + eval yamls + subset GTs |
| OmniDocBench (older) | `data/omnidocbench` | |
| Fox (212) | `data/fox/focus_benchmark_test` | EN+CN page OCR |
| olmOCR-Bench (1403 PDFs, 7 splits) | `data/olmocr_bench/bench_data` | downloaded this session |
| table test set | `data/test_sets/table_test` | |
| **44 uncontrolled captures (Sec 4.6)** | **NOT FOUND** | referenced in `4_experiments.tex:233`; images not in `data/`. → need path |
| **40-page synthetic-defect set (Sec 4.6 / Tab:verified)** | **NOT LOCATED as a dir** | likely generated in-code (defects applied to benchmark pages); need the generator/list |

## 0.6 Thread control + CPU affinity (for the latency curve, 2c)

- onnxruntime threads are code-controlled: `intra_op_num_threads` via
  `_MAIN_ONNX_THREADS` (`models_interface.py`), `onnx_config.py` setter,
  `text_worker.py` `_n_threads`; `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` handled
  in `onnx_config.py` and `rtable_child.py`. → 4/8/16 capping is supported.
- **CPU affinity on Windows**: available via `psutil.Process().cpu_affinity([...])`
  or `start /affinity <mask>`; will pin workers for the thread-count curve.

## 0.7 Vast.ai RTX 6000 Ada

- **No credentials, SSH config, or vast scripts found in the repo.** → **BLOCKED;
  need connection details** (host/port/key or `vast` CLI setup) to run MinerU/Marker
  GPU inference for the accuracy composite (0.4).

---

## What I can do autonomously NOW (no blockers)

1. **olmOCR-Bench PRISM row (2a)** — pilot (arxiv_math, old_scans_math, tables,
   multi_column; 977 PDFs) predictions done; scoring running under WSL (unmodified
   harness; Windows path-sep incompat forced WSL). Full 1403 (all 7 splits) is a
   follow-on prediction run I can launch.
2. **Single-system PRISM composite bootstrap CI** — from existing `odb_full_v20`
   per-page JSONs (the "run-to-run variance at the half-point scale" claim).
3. **Paired bootstrap script (1a)** — build now; feed MinerU per-page once available.
4. **Leave-one-out ablation configs (1c)** — env-flag scripts (0.3).
5. **Thread-capped latency harness (1d)** — prep (0.6).
6. **Mechanical fixes (1f)** — N-column for Tables 11–13 (count from per-page JSONs),
   recover truncated TEDS in Table 9, export real Figure 3 splice crops.
7. Benchmark tables already added this session: olmOCR-Bench (`tab:olmobench`, SOTA
   rows filled, PRISM row pending this run), v1.6 (`tab:v16`), v1.5 (`tab:v15`).

## Decisions (from Phase 0 checkpoint, 2026-07-07)

- **A. MinerU paired run → SKIPPED.** No Vast.ai / no local MinerU. Keep the
  published MinerU `86.47`; **drop the PRISM−MinerU paired bootstrap** and pivot
  the claim to the efficiency frontier + the stable **+5.1 CDM lead**. The
  **single-system PRISM composite CI** (from `odb_full_v20` per-page JSONs) is
  still produced to support the "run-to-run variance at the half-point scale"
  sentence. This makes the "indistinguishable from MinerU" wording unsupportable
  by a paired test — soften it to the efficiency/CDM framing in the paper.
- **B. Sec 4.6 capture set → LOCATED** at `test_images/real/defects/defects-images/`
  (44 images: jpg/png/jpeg). The full-benchmark verified/open/none extension (2f)
  can use these; still HELD per decision D until go-ahead. (The 40-page synthetic
  -defect set for Tab:verified is separate — confirm it is generated in-code.)
- **C. Ablation scope → FIXED SUBSET.** When run, all 5 leave-one-out variants use
  the SAME representative subset; the ablation table is labelled subset-based and
  explicitly NOT comparable to the full Table 9. (Held until go-ahead.)
- **D. Proceed → PILOT ONLY, THEN WAIT.** Report the olmOCR-Bench pilot number and
  hold. No full 1403 run, no ablation runs, no latency runs until reviewed.

## Verdicts landed so far

- Per-page granularity for bootstrap CIs is **available without touching the metric**
  (0.2). The crux statistic is not blocked by harness limitations — only by the
  **absence of a same-harness MinerU full run** (0.4).
- **olmOCR-Bench pilot (2a partial) — MIXED, surfaced honestly.** PRISM v20
  zero-shot, 4 splits, unmodified harness under WSL. table 67.0% (beats MinerU
  60.9 / Marker 57.6 — genuine transfer), multi_column 64.3%, arxiv_math 56.0%,
  old_scans_math 34.9%; 4-split overall 55.5% ± 1.7%. **Tables transfer; math
  does NOT** — PRISM's OmniDocBench CDM lead over MinerU inverts here (arxiv_math
  56.0 vs 75.4) because Texo-20M often emits LaTeX that fails to render in KaTeX
  (the render-exact test defeats the matcher-normalization salvage that lifts
  CDM). This is the orthogonal-metric confirmation of the "CDM lead is partly
  matcher-specific" critique. **Paper implication is a positioning decision (not
  made):** publish the full olmOCR-Bench table honestly (strong tables, weak math)
  or scope it to the table/reading-order strength. Full 1403 run held per D.
  Provenance: `preds/olmocr_pilot/pilot_summary.txt`, `paper.md` log entry.


## olmOCR-Bench full 1403 — LANDED (2026-07-07)
- **PRISM Overall 59.3% ± 1.1%** (raw zero-shot, unmodified harness, commit 54a96a6). Per-cat: AR 56.0 / OSM 34.9 / Tab 67.0 / OS 20.2 / HF 66.6 / MC 64.3 / LTT 69.2 / Base 96.0.
- Wins vs old MinerU 1.3.10: LTT (69.2 vs 39.1), multi-col (64.3 vs 59), tables (67 vs 60.9). Losses: math (recognition-bound), headers_footers (66.6 vs 96.6).
- Math post-process (normalize_v3) = +0.1 Overall → recognition-bound confirmed; report RAW.
- Base is NOT a missing folder: it is the harness BaselineTest (96.0), already included.
- VERSION CAVEAT: baselines are stale (MinerU 1.3.10); current MinerU 2.5.4=75.2. Framed honest/version-labeled, NO SOTA claim (user decision).
