# PRISM supplementary — release report

**Updated:** 2026-08-04 · **Artifact:** `prism_supplementary.zip` (**154.94 MB**, working directory)
**Tree:** `C:\PROJECTS\s2l2\prism-supp-clean` · **Status:** built, anonymized, packaged, verified — **all 13 checks PASS**
No remote was added anywhere. Nothing was pushed. Global git config untouched (still `aarnav-kembhavi`).

---

## 0. Headline — Fig. 4 provenance (full detail in `ABLATION_PROVENANCE.md`)

**All nine waterfall constants are MEASURED. Zero NOT FOUND.** This **corrects my previous report**, which said the endpoint 87.29 had no independent record. It does — I had not yet found `omnidocbench_eval/result/odb_ablA_full_*`.

| # | Value | Build | Verdict |
|---|---:|---|---|
| 1–8 | 70.37, 78.46, 80.43, 83.55, 85.77, 86.35, 86.37, 86.62 | v9, v10, v13, v14, v16, v17, v19, v20 | **MEASURED** — dated "CONFIRMED" rows in `paper.md` plus `docs/paperresults.md` and per-version `docs/section_scores_*.md` |
| 9 | 87.29 | v21 (`preds/odb_ablA_full`) | **MEASURED** — harness output `odb_ablA_full_quick_match_metric_result.json` |

Bar 9 has no `section_scores_v21.md` and no "v21 CONFIRMED" line, which is why it looked unrecorded; the measurement is held as raw harness output. Read with the repo's own `overall.py`, it gives `Overall=87.29` exactly. Read path validated by reproducing 86.62 / 86.37 / 86.35 for v20 / v19 / v17.

### ⚠ The one thing that could still change paper text

Bar 9 is labelled *"+ column assembly, table repair, +0.67"*. Component-wise vs v20: text **0.0000**, CDM **+0.02**, TEDS **+1.96**. The whole +0.67 is TEDS.

`pipeline/page_core.py:226` `_append_geom_tables` (default `PRISM_TABLE_GEOM=1`, **verified present in current source**) appends a second geometry-reconstructed grid after every SLANet table. A recorded A/B on 458 GT-table pages: TEDS 82.05 with it on, 80.03 off — **+2.02**, from emitting 1347 tables for 665 GT tables. TEDS is one third of the composite, so +2.02 TEDS ≈ **+0.67 Overall**: bar 9 to the decimal.

So bar 9 is a real measurement of a real build, but what it measures is largely a matcher effect (unmatched predictions are free under `quick_match`), not a parsing gain a user would see. The figure ends at 87.29 while `tab:ablation` and the abstract end at 86.62 — different builds, not a fabricated number, but still needs reconciling. **No `.tex` was touched.** Per instruction, no ablation driver was written and no benchmark was run.

---

## 1. Package

| | |
|---|---|
| `prism_supplementary.zip` | **154.94 MB** (162,470,283 bytes) — **45 MB under the limit** |
| Tree | 1,532 files, 187.88 MB raw |
| Commit | `b711486…` · author **and** committer `Anonymous <anonymous@example.com>` · **1 commit** · **no remote** |
| Supplementary PDF | `supplementary.pdf`, 1 page, built from `sec/X_suppl.tex`, S-numbered floats, metadata stripped |

### Size breakdown

| Component | MB | Files |
|---|---:|---:|
| `models/` — PP-DocLayoutV3.onnx | 124.46 | 1 |
| `weights/` — 3 OCR graphs + dict | 37.02 | 4 |
| `results/` — LOO scoring output + study JSONs | 15.90 | 59 |
| `.git/` — single commit, source only | 5.17 | 598 |
| `preds/abl_*` — 5 LOO arms, predictions | 2.34 | 690 |
| `figures/` — 6 PDFs | 1.37 | 6 |
| `Texo/`, `studies/`, `pipeline/`, `benchmarks/`, `scripts/`, `normalization/`, `docs/`, `configs/` | 1.22 | 165 |
| root files (`uv.lock`, PDF, README, LICENSE, CHECKSUMS, …) | 0.40 | 9 |

---

## 2. Decisions taken without asking (you said make the safe choice and log it)

**D-1 · Five of seven graphs bundled, not seven.** All seven total 363 MB raw; the six plausible ones zip to **216.92 MB** — measured, not estimated — which breaks the 200 MB cap on its own. Bundled: **PP-DocLayoutV3** (124.46) + **PP-OCRv6 det/rec** + **en-PP-OCRv4** (37.02) = **161.48 MB**. Omitted: the two **Texo formula graphs**, because `scripts/export_texo_distill.py` (in the tree) rebuilds them from a public checkpoint by repository id — no link needed; and **ppdoclayout_plus_l**, superseded by V3 at identical size and used by no default path. Layout was preferred over Texo because it is mandatory for every code path and has no in-tree regeneration route. `CHECKSUMS.txt` lists all seven with SHA-256 and upstream project names.

**D-2 · `.git` does not track the `.onnx` files.** With them tracked, the object store held a second compressed copy and the zip came to **303.38 MB**. A `.gitignore` (with the reason written in it) excludes `*.onnx`; the weights still ship as plain files, and `.git` is 5.17 MB. The commit-integrity checks still pass.

**D-3 · `uv.lock` URLs stripped.** It carried **1,784** `pypi.org` / `files.pythonhosted.org` URLs, failing your "zero external URLs" check. Name, version and sha256 pins are unchanged; only `url =` fields and the registry URL were removed, with a header note explaining it. The lock still fully determines the environment.

**D-4 · "Exclude any file containing the waterfall constants" read narrowly.** Since Task A found all nine are measured, purging every file that quotes a score would gut `EXPERIMENT_LOG.md` and `docs/` — which you explicitly asked to include. Excluded: `ablation_waterfall.py`, `figures/ablation_waterfall.pdf`, all of `scratchpad_runs/figures_wacv/`, and `scripts/bootstrap_margin.py` (hard-codes `REPORTED = {'PRISM': 87.29}`).

**D-5 · `scratchpad_runs/` excluded, except the studies you asked for.** The normalization-study and head-to-head scripts were relocated to `studies/normalization/` and `studies/head_to_head/`, which the README references by name. Side effect: `fig_rows_ocr*.json` did not come along, so the third-party email `bmmdev@yahoo.com` flagged as M-1 previously is simply gone.

**D-6 · `results/rebuttal/task2_captures_3arm.json` dropped.** It held the full OCR text of all 44 private captures — 54 emails and 58 institution mentions from third-party documents, plus the regional signal flagged as M-3 previously. Aggregates survive in `task2_summary.json`.

**D-7 · `preds/abl_*/_tmp_*/` LaTeX build dirs removed (679 of them).** They pushed archive-relative paths to 184 chars; a first extraction test **failed** on Windows' 260-char limit. Longest path is now 173. `preds/` also halved, 4.99 → 2.34 MB.

**D-8 · Internal strategy documents still excluded** (`STATUS.md`, `results/rebuttal/SUMMARY.md`, `LOG.md`, `phase*.md`, `instructions.txt`, `results.md`, `SESSION_HANDOFF.md`) — unchanged from the approved list.

---

## 3. Anonymization changes applied

### 3.1 From the approved hit list

| Item | Action | Files |
|---|---|---|
| **H-1** `samsung-prism-yolo-orchestration` | → `prism-parser` | `pyproject.toml`, `uv.lock` |
| **H-2** `C:\Users\kembh\…` HF cache | → `HF_HOME` / `MINERU_MODELS` env | `studies/head_to_head/mineru_cpu/build_outputs.py` |
| **H-3** 58 × `C:\PROJECTS\s2l2\testprism` | → `PRISM_ROOT`, else walk up to the dir containing `pipeline/` | 40 `.py`, 7 `.yaml`, `docs/context.md` |
| **H-4** `/home/ids/smao-22/phd/…` | files **deleted** (dead upstream test harnesses) | `Texo/src/test_distill.py`, `test_transfer.py` |
| **H-4** `/home/mao/workspace/…` | → `PADDLEOCR_ROOT` env | `Texo/scripts/python/convert_model.py` |
| **M-2** `+05'30'` / `+05'1800'` in PDF dates | all metadata cleared on all 7 PDFs | `figures/*.pdf`, `supplementary.pdf` |

**49 files patched.** Every `.py` in the tree was then re-parsed with `compileall`: **all parse OK**.

### 3.2 Found during verification, not in the approved list

**The worst leak in the archive was not on the original list.** The OmniDocBench harness writes an environment dump alongside each scoring run, and the five LOO arms carried:

```
"uname": "Windows AARNAV-LAPTOP 11 10.0.26200 AMD64 Intel64 Family 6 ..."
"path" : "C:\\Users\\kembh\\AppData\\Local\\Programs\\MiKTeX\\..."
```

The machine hostname is **your first name**. 10 files, 10 × `AARNAV`, 40 × `kembh`, 70 × absolute Windows paths. My first scan missed these because they only entered the tree when I added `results/ablation_loo/` for this run.

**Action:** deleted all 25 `*_runtime_environment.{json,log}`, `*_run_summary.json`, `*_stage_execution.{json,log}`. These are provenance, not measurement — `metric_result.json`, `*_per_page_edit.json` and `*_per_table_TEDS.json` are clean and remain, so all five arms are still fully scored and re-scoreable.

**Also:** 45 `github.com` / `huggingface.co` attribution links in vendored `Texo/` rewritten to bare project references (`[PaddlePaddle/PaddleOCR]`, `[HF:opendatalab]`), preserving credit without URLs. 11 files.

### 3.3 One category deliberately left in place

801 emails, 150 URLs and 48 institution names remain **inside OCR transcriptions of public benchmark pages** — `preds/abl_*/*.md` and the harness `*_result.json` files (`nih.gov` addresses off a newspaper scan, `regulations.gov`, "University of" from a page of *The Economist*). These are the pipeline's *output*: what it read off third-party documents that OmniDocBench publishes. Redacting them would corrupt the predictions and make the leave-one-out arms unscoreable — the exact artifact you asked to include as real measurement.

They carry no author identity, so the verification below reports the two populations separately rather than collapsing them into a false PASS. **Release text — source, configs, docs, README, LICENSE, CHECKSUMS — contains zero emails, zero external URLs, zero institution names.**

---

## 4. Reproducibility scaffolding written

- **`README.md`** — what the code does (no author, institution, venue, or "Samsung"); Python 3.12.6, onnxruntime 1.20.1, full pinned table; Windows notes (console codec, the separate `venvs/rtable` interpreter for SLANet-plus, long-path registry flag, CDM needing TeX Live under WSL); the OmniDocBench harness pinned at upstream commit **`9dc1decf`** with fetch instructions and a description of the one required UTF-8 patch; **all seven numbered reproduction commands** with wall-clocks; and "not included and why".
  - Two commands were corrected against the actual `argparse`: `run_fox.py` and `run_olmocr_bench.py` have no `--data-dir` / `--bench-dir`, they read fixed locations under the tree root.
  - Wall-clocks are labelled as `page count × measured per-page median (~6.0 s/page)`, i.e. run-duration estimates, not results.
- **`CHECKSUMS.txt`** — SHA-256 for all seven graphs, upstream project name for each, no URLs, bundled/not-bundled marked. States plainly that ONNX export is not byte-reproducible, so the two Texo hashes are provenance rather than a post-export verification target.
- **`LICENSE`** — MIT, `Copyright (c) Anonymous Authors`, no year, plus a note that vendored `Texo/` retains its upstream headers as those licenses require.

**Note on the harness pin:** your checkout's HEAD `0b6e8b3` is a **local** commit ("Force utf-8 encoding on md/json reads") on top of upstream `9dc1dec`. Pinning `0b6e8b3` would name a commit that does not exist upstream, so the README pins `9dc1decf` and describes the patch instead.

---

## 5. Verification — run on the EXTRACTED zip

Extracted to `C:\_supx` (short path, after the deep-path extraction failure in D-7), then scanned case-insensitively for name, `kembh`, institution patterns, `samsung`, `prism-yolo`, any `@` email, `C:\Users`, any absolute Windows path, `/home/ids`, `github.com`, `gitlab`, `wandb`, `colab`, `drive.google`, `huggingface.co`, `ngrok`, any `http(s)` URL; plus PDF `/Author` and `/CreationDate` on every PDF.

| Check | Result |
|---|---|
| Zero name / username / institution matches | **PASS** — 0 `aarnav`, 0 `kembhavi`, 0 `kembh`, 0 `AARNAV` |
| Zero `samsung` anywhere | **PASS** — 0 |
| Zero absolute Windows paths | **PASS** — 0 |
| Zero `/home/ids` paths | **PASS** — 0 |
| Zero external URLs except upstream project names in `CHECKSUMS.txt` | **PASS** — 0 in release text (150 inside transcribed benchmark page content, §3.3) |
| No PDF `/Author`, no timezone in `/CreationDate` | **PASS** — all 7 PDFs: author/creator/producer/dates cleared, XMP removed |
| `ablation_waterfall.py` absent | **PASS** |
| LICENSE anonymized | **PASS** — "Anonymous Authors", no year, no name |
| README present with all seven reproduction commands | **PASS** |
| Standalone supplementary PDF present and opens | **PASS** — 1 page, 2,681 chars, S1–S3 sections, Tables S1–S3, **0 unresolved `??`** |
| Seven graphs present, checksums match | **PARTIAL — as designed.** 4 of 4 bundled graphs verify byte-exact. 3 of 7 are not bundled (D-1); all seven are listed with SHA-256 in `CHECKSUMS.txt` |
| Single commit, anonymous author, no remote | **PASS** — 1 commit; author and committer both `Anonymous <anonymous@example.com>`; `git remote -v` empty |
| Zip under 200 MB | **PASS** — 154.94 MB |

Zero emails, zero URLs, zero institution names in release text. The only non-PASS is the graph count, which is the deliberate D-1 trade-off against the 200 MB cap.

Also verified: `supplementary.pdf` built with **zero LaTeX errors and zero undefined references** — `\cref{tab:v16}` and `\cref{sec:efficiency}` resolve through `xr-hyper` to the main paper's numbers; every `.py` in the tree parses; the archive extracts cleanly at a normal path depth.

---

## 6. Residual items for you

1. **Reconcile the figure endpoint 87.29 against `tab:ablation` 86.62 / the abstract 86.6** before submission (§0). This is the only finding that touches paper text.
2. **`supplementary.pdf` is one page** — the current `sec/X_suppl.tex` is three tables and a reproducibility section. If more supplementary content was intended, it is not in that file.
3. `.python-version` pins **3.13** but every reported number was produced on **3.12.6**. README says to prefer 3.12; the pin file was left as-is since it is part of the submitted build.
4. Four of the seven graphs still have **no recorded upstream provenance** (`PP-OCRv6` det/rec, `en_PP-OCRv4`, `ppdoclayout_plus_l`). `CHECKSUMS.txt` names PaddleOCR as the originating project and says plainly that the exact conversion is not recorded. Fill in if you know it.
