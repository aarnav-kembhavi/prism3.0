# PRISM — Restructure Plan (Phase 1: audit only)

**Generated:** 2026-08-24 · Branch `latency-hardening` @ `d91c6b1` (ahead 1 of origin)
**Scope:** Read-only audit. No file was moved, deleted, renamed or modified. This
document is the only thing written.

**Frozen (not touched, not proposed for movement):** `prism_supplementary.zip`,
`paper_overleaf.zip`, `paper/`, `paper_overleaf/`, everything under `figures/`.
They are analysed below where the brief asks for it (duplication), but every
proposal leaves them exactly where they are.

---

## 0. Headline findings

Read these before anything else.

### 0.1 🔴 `weights/` IS TRACKED IN GIT — via Git LFS

`weights/` is **not** in `.gitignore`. Four files are tracked:

| File | Size | Storage |
|---|---|---|
| `weights/PP-OCRv6_det_small.onnx` | 9.9 MB | **Git LFS** |
| `weights/PP-OCRv6_rec_small.onnx` | 21.2 MB | **Git LFS** |
| `weights/en_PP-OCRv4_rec.onnx` | 7.7 MB | **Git LFS** |
| `weights/en_dict.txt` | 190 B | plain blob |

`.gitattributes` routes `*.onnx` and `*.safetensors` through LFS, and
`git lfs ls-files` confirms all three ONNX files are real LFS pointers — so the
repo is not carrying 39 MB of binary in its object history. **But a teammate who
clones without `git lfs install` gets 130-byte pointer text files where the OCR
models should be, and the pipeline will fail at model load with an opaque
ONNX Runtime parse error.** This is the single most likely first-run failure.

### 0.2 🔴 `venvs/` is NOT tracked — and neither are the models that matter

`venvs/` and `.venv_rtable/` are both gitignored: **0 tracked files**, 10.0 GB and
1.5 GB on disk respectively. Good — but see §7.3, because `.venv_rtable` is a
**runtime** dependency of the table path, not a dev convenience.

`models/` is gitignored too — **0 tracked files, 510 MB on disk**. The layout
detector the pipeline cannot start without (`models/ppdoclayout/ppdoclayout_plus_l.onnx`)
is **not in the repo and not documented as a download**. See §7.1.

### 0.3 🔴 `omnidocbench_eval` is a broken submodule

`git ls-files -s omnidocbench_eval` returns mode `160000` (a gitlink) pointing at
commit `0b6e8b3fe09aa486db521cf2d1cb2783ca2157e2` — **but there is no
`.gitmodules` file.** A fresh clone produces an empty `omnidocbench_eval/`
directory and no way to populate it. Git will not even report it as missing.

Recoverable: the nested repo's `origin` is `https://github.com/opendatalab/OmniDocBench.git`
and its local HEAD matches the gitlink exactly. The fix is a `.gitmodules` entry —
but nobody can guess that from the repo alone. `README.md` calls it a "submodule",
which is the only surviving hint.

### 0.4 🔴 2,182 score artifacts backing the paper exist on this machine only

`omnidocbench_eval/result/` holds 2,182 files — every metric JSON behind every
number in `paper/`, `docs/paperresults.md` and `docs/section_scores_odb_full_v*.md`.

They are:
- **not** in the parent repo (a gitlink stores only a commit SHA, never working-tree files), and
- **not** in the nested repo either — `omnidocbench_eval/.gitignore:29` is `result/*`.

`git status` is clean in both repos. **These 1.4 GB of results are one disk
failure from gone, and they are the provenance for the conference submission.**
Back them up before any restructure begins.

### 0.5 🟠 Uncommitted work is substantial

`results/` (2.7 GB, 659 files) is neither tracked nor gitignored. `scripts/rebuttal/`
(11 scripts) is untracked. `RELEASE_REPORT.md`, `findings.md`, `results.md`,
`ABLATION_PROVENANCE.md` are untracked. Two tracked files are deleted in the working
tree but not committed (`pipeline/tatr_worker_onnx.py`, `scripts/export_tatr_onnx.py`).

Commit or explicitly quarantine this before moving anything.

---

## 1. INVENTORY

Classification is derived from actual imports (AST parse of all 134 first-party
`.py` files) and from path string literals in code — not from filenames.

Legend: **RUNTIME** = reachable from `pipeline/orchestrate.py` or `app.py` ·
**DEV** = eval/benchmark/scratch only · **PAPER** = submission artifact ·
**DEAD** = nothing in the repo references it.

### 1.1 Code

| Path | Size / files | Class | Referenced by (evidence) |
|---|---|---|---|
| `pipeline/` | 0.6 MB, 14 `.py` | **RUNTIME** | Entry module. All 14 modules have at least one referrer; none dead. See §1.6. |
| `normalization/` | 0.1 MB, 6 `.py` | **RUNTIME** | `pipeline/orchestrate.py:*`, `benchmarks/run_omnidocbench.py`, 8 `scripts/rebuttal/*` |
| `app.py` | 7.2 KB | **RUNTIME** | FastAPI UI. Spawns `pipeline/orchestrate.py` as a subprocess (`app.py:65`). |
| `web/` | 1 file (`index.html`) | **RUNTIME** | `app.py:126` → `FileResponse(ROOT / "web" / "index.html")` |
| `benchmarks/` | 405 MB, 1486 files | **DEV** | Eval harness. 37 files tracked; the remaining ~1449 are gitignored `preds_*`/`mineru_*`/`formula_eval` run artifacts. |
| `scripts/` | 0.1 MB, 15 files | **DEV** | Only 3 of 15 tracked. `scripts/rebuttal/` (11 files) entirely untracked. |
| `Texo/` | 296 MB, 260 files | **RUNTIME** (partly) | Vendored third-party repo (own LICENSE/pyproject/uv.lock/.gitattributes). Runtime needs **only** `Texo/model/` (`math_worker_onnx.py:696`) and `Texo/src/` (`models_interface.py:69`). 222 files tracked; `Texo/model/` is gitignored. |
| `omnidocbench_eval/` | 1.4 GB, 2467 files | **DEV** + **PAPER** | Gitlink, no `.gitmodules` (§0.3). Code = DEV (`benchmarks/run_omnidocbench.py:47`); `result/` = PAPER provenance (§0.4). |

### 1.2 Models and weights

| Path | Size | Class | Referenced by |
|---|---|---|---|
| `weights/PP-OCRv6_det_small.onnx` | 9.9 MB | **RUNTIME** | `models_interface.py:133`, `text_worker.py:322`, `normalization/verified.py:32` |
| `weights/PP-OCRv6_rec_small.onnx` | 21.2 MB | **RUNTIME** | `models_interface.py:134`, `text_worker.py:323` |
| `weights/en_PP-OCRv4_rec.onnx` | 7.7 MB | **RUNTIME** | `models_interface.py:135`, `text_worker.py:331` |
| `weights/en_dict.txt` | 190 B | **RUNTIME** | `models_interface.py:136`, `text_worker.py:332` |
| `models/ppdoclayout/ppdoclayout_plus_l.onnx` | 129.7 MB | **RUNTIME** | `models_interface.py:197`, `orchestrate.py:78`, `run_omnidocbench.py:57` |
| `models/ppdoclayoutv3/PP-DocLayoutV3.onnx` | 130.5 MB | **RUNTIME** | `models_interface.py:200` (gated on `PRISM_PPDL_V3`) |
| `models/doclayout_yolo_docstructbench_imgsz1024.onnx` | 75.3 MB | **DEAD** | Zero hits for `doclayout_yolo` in any `.py`. Legacy detector, replaced by PP-DocLayout. |
| `models/MFD/YOLO/yolo_v8_ft_640_dyn.onnx` | 174.5 MB | **DEAD** | Zero hits for `yolo_v8_ft`. `run_omnidocbench.py:54` comment confirms MFD was *replaced*. |
| `models/.cache/huggingface/` | ~0 | **DEAD** | HF download metadata stubs. |
| `Texo/model/onnx/*` | ~76 MB | **RUNTIME** | `math_worker_onnx.py:713,718` (`encoder_model.onnx`, `decoder_model_merged.onnx`) |

**~250 MB of `models/` is dead weight.** This also reconciles the README's
"~283 MB of weights" claim: that figure counts only the live set
(PP-DocLayoutV3 130 + Texo ~76 + PP-OCRv6 31 + en_v4 8 + SLANet-plus 7.4 inside
the child venv ≈ 283 MB), not the 510 MB actually sitting in `models/`.

### 1.3 Data and outputs

| Path | Size / files | Class | Notes |
|---|---|---|---|
| `data/` | 5.5 GB, 19173 files | **DEV** | Gitignored, 0 tracked. Five benchmark corpora: `omnidocbench/` (1971), `omnidocbench_full/` (3429), `olmocr_bench/` (9440), `fox/` (960), `test_sets/table_test` (16). None present after a clone. |
| `preds/` | **16.6 GB**, 79400 files | **DEV** | Gitignored. 113 named benchmark runs (`ab_*`, `odb_full_v*`, `_mineru_raw`, …). Written by `benchmarks/*` (`run_fox.py:103`, `rerun_mixed.py:13`, `benchmark_glare.py:36`, `run_olmocr_bench.py:111`). Largest single directory in the repo. |
| `outputs/` | 100 MB, 607 files | **DEV** (runtime product) | Gitignored. Per-image pipeline output `<stem>_output/{main.tex,main.pdf,assets/,logs/}`. Written by `orchestrate.py:213` and `app.py:74`. 95 stale run directories. |
| `results/` | **2.7 GB**, 659 files | **DEV** | **Not tracked and not gitignored.** Rebuttal experiment output required by `instructions.txt` ("Log every command to results/rebuttal/LOG.md"). Genuinely uncommitted work. |
| `test_images/` | 311 MB, **293 tracked** | **DEV** | Fully tracked, not gitignored. `real/`, `synthetic/`, `crops/`, `rotation_benchmark/`. `real/defects/defects-images/` (44 captures) is cited by the paper's Sec 4.6 and used by 4 `scripts/rebuttal/*`. |
| `scratchpad_runs/` | 738 MB, 1182 files | **DEV** | Gitignored. 15 sub-experiments. **Contains the paper's figure generators** — see §7.5. |
| `_web_uploads/` | 0.4 MB, 1 file | **DEAD** (transient) | Gitignored. `app.py:25` recreates it. One orphaned upload left behind by a crashed job. |
| `.cache/huggingface/` | ~0 | **DEAD** | Stray HF cache at repo root. |
| `__pycache__/`, `pipeline/__pycache__/`, … | — | **DEAD** | Gitignored. Includes an orphan: `pipeline/__pycache__/tatr_worker_onnx.cpython-312.pyc` with no corresponding `.py`. |

### 1.4 Paper artifacts (frozen — inventory only)

| Path | Size / files | Class | Notes |
|---|---|---|---|
| `paper/` | 0.7 MB, 28 files (19 tracked) | **PAPER** | Authoritative LaTeX source. 9 untracked files are gitignored build artifacts (`main.aux/.bbl/.log/.pdf/…`). |
| `paper_overleaf/` | 0.2 MB, 19 files (19 tracked) | **PAPER** | Overleaf export. Behind `paper/` — see §4.1. |
| `paper_overleaf.zip` | 94 KB | **PAPER** | Snapshot of `paper_overleaf/`. Stale — see §4.1. |
| `figures/` | 18.1 MB, 28 files (**5 tracked**) | **PAPER** | 5 tracked PDFs; `figures/*.png` gitignored; `multilingual/`, `normalisation_verified.*`, `qualitative_io.pdf`, `thread_scaling.pdf`, `S2L (23).pdf` untracked. **None of these is `\includegraphics`'d by `paper/` or `paper_overleaf/`** — see §4.4. |
| `prism_supplementary.zip` | 159.5 MB | **PAPER** | Anonymised code release. Contains an embedded `.git/` (564 objects) — checked: single initial commit, author `Anonymous <anonymous@example.com>`, message "Anonymous supplementary code release". **No de-anonymisation risk.** (Only residue is a `+0530` timezone offset in the commit log.) |
| `prism_supplementary_clean/` | 162.5 MB, 64 files | **PAPER** (working copy) | Untracked staging tree for the above — see §4.5. |
| `S2L (23).pdf`, `S2L (24).pdf` | 2.8 MB / 0.9 MB | **PAPER** | Untracked compiled submission PDFs at repo root. `S2L (23).pdf` is **byte-identical** (md5 `c47723bd…`) to `figures/S2L (23).pdf`. |
| `paper.md` | 674 lines | **PAPER** | Experiment log. Linked from README. Tracked. |

### 1.5 Docs, config, tooling

| Path | Class | Notes |
|---|---|---|
| `README.md` | **needed** | Entry doc. Contains one broken link — `docs/formula_fix_v2.md` does not exist. |
| `docs/` (10 files) | **needed** | `context.md`, `paperresults.md`, `pipeline_audit_2026-07-04.md`, 4× `section_scores_odb_full_v*.md`, `weakness_analysis_v15.md`, 2 PNGs. All tracked. |
| `pyproject.toml` | **RUNTIME** | Declares deps. Substantially wrong — see §7.2. |
| `uv.lock` | **DEV** | 563 KB. Locks the pyproject set, including the unnecessary torch stack. |
| `.python-version` | **conflicting** | Says `3.13`. `pyproject` requires `>=3.12`; every venv on disk is **3.12.6**. |
| `.gitignore` / `.gitattributes` | **needed** | See §0.1. |
| `STATUS.md` | **DEV** | Phase-0 inventory for a *previous* restructure attempt (branch `wacv-results-hardening`). Tracked, modified. |
| `SESSION_HANDOFF.md` | **DEV** | Handoff note for a superseded branch/session (2026-07-07). Tracked. Stale. |
| `instructions.txt` | **DEV** | Task brief driving `scripts/rebuttal/` + `results/rebuttal/`. Tracked. |
| `paperresults.md` (root) | **DEAD** (superseded) | Stale v1.5 / 11-system comparison. Superseded by `docs/paperresults.md` (v19), which is what README links. See §4.6. |
| `RELEASE_REPORT.md`, `findings.md`, `results.md`, `ABLATION_PROVENANCE.md` | **PAPER** (untracked) | Provenance/analysis reports produced during the submission push. |
| `skill.md` | **DEAD** | An Anthropic **`frontend-design`** skill definition. Nothing to do with PRISM. Tracked. See §6. |
| `.vscode/settings.json` | **DEAD** | Auto-approve rule for `test_rapid_ocr.py` — a file that does not exist. Tracked (despite `.vscode/` being in `.gitignore`; it predates the rule). |
| `.claude/settings.local.json` | **DEV** | Local agent config. Gitignored. |
| `.venv_rtable/` | **RUNTIME** ⚠ | 1.5 GB. Gitignored. Hosts the RapidTable child process — see §7.3. |
| `venvs/` (5 venvs) | **DEV** | 10.0 GB: `gpu`, `mineru_cpu`, `ppocr`, `smol`, `docling_rebuttal`. Gitignored. |

### 1.6 `pipeline/` module-by-module (all live)

| Module | Referrers |
|---|---|
| `orchestrate.py` | `app.py:65` (subprocess), `page_core.py`, `run_omnidocbench.py` |
| `page_core.py` | `orchestrate.py:66`, `models_interface.py`, `formula_v2.py`, `run_omnidocbench.py` |
| `models_interface.py` | `orchestrate.py`, `text_worker.py`, `run_omnidocbench.py`, 4× `scratchpad_runs/*` |
| `text_worker.py` | `orchestrate.py`, `page_core.py`, `run_omnidocbench.py` |
| `math_worker_onnx.py` | `orchestrate.py`, `run_omnidocbench.py`, `benchmarks/compare/run_texo_formulas.py` |
| `formula_v2.py` | `page_core.py:914` (lazy), `detection_postprocess.py`, `ppdoclayout_onnx.py` |
| `detection_postprocess.py` | `orchestrate.py:65`, `formula_v2.py`, `run_omnidocbench.py` |
| `layout_utils.py` | `orchestrate.py:63`, `page_core.py`, `detection_postprocess.py`, `run_omnidocbench.py` |
| `latex_builder.py` | `orchestrate.py:64`, `page_core.py`, `run_omnidocbench.py` |
| `tex_to_md.py` | `latex_builder.py`, `page_core.py`, `run_omnidocbench.py` |
| `onnx_config.py` | 5 pipeline modules + `run_omnidocbench.py` + `scripts/export_texo_distill.py` |
| `ppdoclayout_onnx.py` | `models_interface.py:8`, `scripts/validate_ppdoclayout_onnx.py` |
| `rtable_worker.py` | `page_core.py:85` (lazy), gated on `PRISM_RTABLE` |
| `rtable_child.py` | Spawned by `rtable_worker.py:53` **in a different interpreter** |
| `__init__.py` | package marker |

Nothing in `pipeline/` is dead.

---

## 2. ENTRYPOINTS

18 files define `__main__`. Grouped by who would actually invoke them.

### 2.1 Product entrypoints

**A. `python pipeline/orchestrate.py <image>`** — the pipeline. The one command in the README.

Import closure (module-level, then lazy):
```
orchestrate → onnx_config, detection_postprocess → layout_utils
            → page_core → latex_builder → tex_to_md
                        → text_worker → onnx_config, models_interface
                        → rtable_worker → [subprocess] rtable_child   (PRISM_RTABLE)
                        → formula_v2                                   (lazy, :914)
            → models_interface → ppdoclayout_onnx, page_core
            → math_worker_onnx → onnx_config
            → normalization → pipeline, modality, geometric, frequency_filter, verified
            → evaluation.profiler                          ⚠ DOES NOT EXIST (:69)
```
Directories actually touched: `pipeline/`, `normalization/`, `weights/`,
`models/ppdoclayout/` (+ `models/ppdoclayoutv3/` if `PRISM_PPDL_V3`), `Texo/model/`,
`Texo/src/`, `.venv_rtable/` (or `venvs/rtable/`), writes `outputs/<stem>_output/`.

**B. `python app.py`** — FastAPI UI on `0.0.0.0:8000`.

Does **not** import the pipeline. It shells out:
`subprocess.run([sys.executable, ROOT/"pipeline"/"orchestrate.py", image], env={**os.environ, "PRISM_VISUAL_FIDELITY": "1"})`.
Touches `web/index.html`, `_web_uploads/`, `outputs/`, plus everything in (A).
Additionally requires **`xelatex` or `pdflatex` on `PATH`** (`app.py:87`; picks
`xelatex` when the `.tex` contains `\usepackage{xeCJK}`).

### 2.2 Benchmark entrypoints (DEV)

| Entrypoint | Dirs touched |
|---|---|
| `benchmarks/run_omnidocbench.py` | Full (A) closure + `omnidocbench_eval/` (`:47`), `data/omnidocbench*`, writes `preds/omnidocbench` (`:49`) |
| `benchmarks/run_fox.py` | imports `run_omnidocbench`; `data/fox/`, `preds/fox` |
| `benchmarks/run_olmocr_bench.py` | imports `run_omnidocbench`; `data/olmocr_bench/`, `preds/olmocr_stage`; **shells `pdftoppm`** (poppler) at `:84` |
| `benchmarks/benchmark_glare.py` | imports `run_omnidocbench`; `preds/glare_bench` |
| `benchmarks/rerun_mixed.py` | imports `run_omnidocbench`; `preds/omnidocbench` |
| `benchmarks/make_report.py` | reads eval JSON, writes report MD |
| `benchmarks/olmo_normalize.py` | text normalisation utility |
| `benchmarks/compare/run_{mineru,ppstructure,docling_vlm,smoldocling,gpu_vlm,texo_formulas,ppfn_formulas}.py` | External-baseline runners; each needs a *different* venv (§7.4) |
| `benchmarks/compare/{collect_metrics,measure_external,surya_layout_diag,extract_formula_crops,ppdl_build_cache,ppdl_build_full_cache,probe_apis}.py` | Ad-hoc measurement tools |

`benchmarks/compare/*` import `bench_metrics` as a **top-level module name**, which
only resolves because those scripts `sys.path.insert` the hardcoded absolute repo
path (§3.1) or are run with `benchmarks/compare` as cwd.

### 2.3 Script entrypoints (DEV)

- `scripts/validate_ppdoclayout_onnx.py` — imports `pipeline.ppdoclayout_onnx`
- `scripts/export_texo_distill.py` — imports `pipeline.onnx_config`
- `scripts/bootstrap_margin.py`, `scripts/build_panel_mixed.py` — untracked; figure/CI stats
- `scripts/rebuttal/*.py` (11 files, all untracked) — import `normalization.*`;
  read `test_images/real/defects/defects-images/`; write `results/rebuttal/`

### 2.4 Notebooks

**None** outside the vendored `Texo/` (`Texo/demo.ipynb`, `Texo/TechnoSelection/*.ipynb`,
5 total). No first-party notebook entrypoint exists.

---

## 3. HARDCODED PATHS

### 3.1 Absolute Windows paths in first-party `.py` — file:line

Every one of these is `C:\PROJECTS\s2l2\testprism`. All break on clone.

| File:line | Literal |
|---|---|
| `benchmarks/compare/extract_formula_crops.py:6` | `ROOT = r"C:\PROJECTS\s2l2\testprism"` |
| `benchmarks/compare/ppdl_build_cache.py:9` | `sys.path.insert(0, r"C:\PROJECTS\s2l2\testprism")` |
| `benchmarks/compare/ppdl_build_cache.py:10` | `os.chdir(r"C:\PROJECTS\s2l2\testprism")` |
| `benchmarks/compare/ppdl_build_full_cache.py:9` | `sys.path.insert(0, …)` |
| `benchmarks/compare/ppdl_build_full_cache.py:10` | `os.chdir(…)` |
| `benchmarks/compare/run_gpu_vlm.py:19` | `ROOT = r"C:\PROJECTS\s2l2\testprism"` |
| `benchmarks/compare/run_mineru.py:5` | `sys.path.insert(0, os.path.join(r"C:\PROJECTS\s2l2\testprism", "benchmarks", "compare"))` |
| `benchmarks/compare/run_mineru.py:6` | `os.chdir(r"C:\PROJECTS\s2l2\testprism")` |
| `benchmarks/compare/run_mineru.py:14` | `MINERU = r"C:\PROJECTS\s2l2\testprism\.venv_mineru\Scripts\mineru.exe"` ⚠ **`.venv_mineru` does not exist** (it is `venvs/mineru_cpu`) |
| `benchmarks/compare/run_ppfn_formulas.py:5` | `ROOT = r"C:\PROJECTS\s2l2\testprism"` |
| `benchmarks/compare/run_texo_formulas.py:3` | `sys.path.insert(0, …)` |
| `benchmarks/compare/run_texo_formulas.py:4` | `os.chdir(…)` |
| `benchmarks/compare/surya_layout_diag.py:7` | `sys.path.insert(0, …)` |
| `benchmarks/compare/surya_layout_diag.py:8` | `os.chdir(…)` |
| `scripts/bootstrap_margin.py:31` | `ROOT = r'C:\PROJECTS\s2l2\testprism'` |
| `scripts/build_panel_mixed.py:16` | `CMP = r"C:\PROJECTS\s2l2\testprism\outputs\c06_output\main.pdf"` |
| `scripts/build_panel_mixed.py:80` | `FIGDIR = r"C:\PROJECTS\s2l2\testprism\figures\multilingual"` |
| `scripts/validate_ppdoclayout_onnx.py:6` | `sys.path.insert(0, …)` |
| `scripts/validate_ppdoclayout_onnx.py:7` | `os.chdir(…)` |
| `scripts/rebuttal/phase2_modality_audit.py:20` | `ROOT = r'C:\PROJECTS\s2l2\testprism'` |
| `scripts/rebuttal/phase2b_fp_bitidentity.py:10` | `ROOT = r'C:\PROJECTS\s2l2\testprism'` |
| `scripts/rebuttal/phase3_acceptance.py:11` | `ROOT = r'C:\PROJECTS\s2l2\testprism'` |
| `scripts/rebuttal/phase3_aggregate.py:11` | `ROOT = r'C:\PROJECTS\s2l2\testprism'` |
| `scripts/rebuttal/phase4_docling.py:19` | `ROOT = r'C:\PROJECTS\s2l2\testprism'` |
| `scripts/rebuttal/phase5_aggregate.py:12` | `ROOT = r'C:\PROJECTS\s2l2\testprism'` |
| `scripts/rebuttal/phase5_docunet.py:23` | `ROOT = r'C:\PROJECTS\s2l2\testprism'` |
| `scripts/rebuttal/sweep_threshold.py:27` | `ROOT = r'C:\PROJECTS\s2l2\testprism'` |
| `scripts/rebuttal/task2_capture_arms.py:26` | `ROOT = r'C:\PROJECTS\s2l2\testprism'` |
| `scripts/rebuttal/task2_score.py:25` | `ROOT = r'C:\PROJECTS\s2l2\testprism'` |

**29 sites across 20 files.** All in `benchmarks/compare/` and `scripts/` — **zero in
`pipeline/` or `normalization/`.** The runtime is clean; the tooling is not.

### 3.2 Absolute paths in generated YAML (regenerable, gitignored)

24 further sites in `benchmarks/compare/*/eval_config.yaml` and
`benchmarks/compare/preds_*/eval_config.yaml` (`data_path: C:\PROJECTS\s2l2\testprism\…`,
lines 5 and 10 of each). These are **emitted by the eval harness**, not authored;
they regenerate on each run. Do not hand-edit — fix the emitter.

### 3.3 Repo-root-relative paths in RUNTIME code (the migration hazard)

The runtime resolves everything from a computed root:

| File:line | Definition |
|---|---|
| `pipeline/math_worker_onnx.py:32` | `ROOT_DIR = dirname(dirname(abspath(__file__)))` |
| `pipeline/models_interface.py:68` | same |
| `pipeline/rtable_worker.py:21` | same |
| `pipeline/text_worker.py:34` | same |
| `normalization/verified.py:30` | same |
| `pipeline/orchestrate.py:32` | `_ROOT = Path(__file__).resolve().parent.parent` |

`dirname(dirname(__file__))` means *"the directory containing my package"*. Consumers:

| File:line | Path built |
|---|---|
| `models_interface.py:69` | `ROOT_DIR/Texo/src` → `sys.path.append` |
| `models_interface.py:70` | `ROOT_DIR/text-table-latex` → ⚠ **does not exist** |
| `models_interface.py:133-136` | `ROOT_DIR/weights/{PP-OCRv6_det_small,PP-OCRv6_rec_small,en_PP-OCRv4_rec}.onnx`, `en_dict.txt` |
| `models_interface.py:184` | `ROOT_DIR/Texo/model` |
| `models_interface.py:197` | `ROOT_DIR/models/ppdoclayout/ppdoclayout_plus_l.onnx` |
| `models_interface.py:200` | `ROOT_DIR/models/ppdoclayoutv3/PP-DocLayoutV3.onnx` |
| `math_worker_onnx.py:696` | `ROOT_DIR/Texo/model` (→ `/onnx/encoder_model.onnx`, `/onnx/decoder_model_merged.onnx`) |
| `math_worker_onnx.py:37`, `text_worker.py:39` | inject `ROOT_DIR` into child `PYTHONPATH` |
| `text_worker.py:322-332` | `ROOT_DIR/weights/*` |
| `normalization/verified.py:32` | `ROOT_DIR/weights/PP-OCRv6_det_small.onnx` |
| `rtable_worker.py:23` | `ROOT_DIR/pipeline/rtable_child.py` |
| `rtable_worker.py:27-30` | `ROOT_DIR/venvs/rtable/Scripts/python.exe`, else `ROOT_DIR/.venv_rtable/Scripts/python.exe` |
| `rtable_worker.py:58` | child `cwd=ROOT_DIR` |
| `orchestrate.py:78` | `_ROOT/models/ppdoclayout/ppdoclayout_plus_l.onnx` |
| `orchestrate.py:213` | `_ROOT/outputs/<stem>_output` |
| `run_omnidocbench.py:47,49,57` | `ROOT/omnidocbench_eval`, `ROOT/preds/omnidocbench`, `ROOT/models/ppdoclayout/…` |

> ⚠ **This is why `pipeline/` cannot simply be moved to `src/pipeline/`.**
> `dirname(dirname(__file__))` would resolve to `src/`, and every model path above
> would break at once — with a `RuntimeError: PP-DocLayout model missing`
> (`orchestrate.py:102`) if you're lucky, and a silent OCR-disabled path if not.
> Phase 2 must introduce a single explicit root resolver **before** any move. See §5.1.

### 3.4 Other literals

- `pipeline/orchestrate.py:333` — `"assets/figure_header_logo.png"` (relative to the emitted output dir; correct)
- `pipeline/page_core.py:70-72` — rewrites `{figure_NNN}` → `{assets/figures/figure_NNN}` (output-relative; correct)
- `benchmarks/run_omnidocbench.py:47-49` — `omnidocbench_eval/demo_data/omnidocbench_demo/{images,OmniDocBench_demo.json}` defaults
- `scripts/rebuttal/{phase2_modality_audit:32, phase3_acceptance:22, sweep_threshold:42, task2_capture_arms:36}` — `test_images/real/defects/defects-images`
- No POSIX absolute paths (`/mnt`, `/home`, `/usr`) anywhere in first-party code.

---

## 4. DUPLICATION

### 4.1 `paper/` vs `paper_overleaf/` vs `paper_overleaf.zip` — **`paper/` is authoritative**

Full recursive diff, `paper/` vs `paper_overleaf/`:
- **Only in `paper/`:** `main.aux`, `main.bbl`, `main.blg`, `main.brf`, `main.fdb_latexmk`, `main.fls`, `main.log`, `main.out`, `main.pdf` — all gitignored LaTeX build artifacts.
- **Content difference:** exactly one file, `sec/4_experiments.tex` (324 lines vs 286).
- **All 18 other files are byte-identical.**

`paper/sec/4_experiments.tex` contains **two tables that `paper_overleaf/` lacks**:
- `\label{tab:perf_threads}` — "Latency vs. thread budget (Table 4b)", 4/8/16-core rows (19 lines)
- `\label{tab:loo}` — leave-one-out ablation on the 136-page subset (19 lines)

This matches the commit history: `paper_overleaf` was exported at `268c401`, then
`cf34af5` ("Latency Table 4b: thread-scaling rows") and `5a60c24` ("add leave-one-out
ablation table (tab:loo)") landed in `paper/` only.

**Verdict: `paper/` is ahead. `paper_overleaf/` is a stale export missing two
results tables. Do not treat the Overleaf copy as newer because it was
"exported later".**

`paper_overleaf.zip` vs `paper_overleaf/`: 19 entries, **18 byte-identical**;
`sec/4_experiments.tex` differs (zip 25,259 B vs dir 25,065 B — the directory has
since been edited *down*, and shows as `M` in `git status`). So the zip is a
snapshot of a slightly *older, larger* `paper_overleaf/`. Three-way drift:
`paper/` (+2 tables) → `paper_overleaf.zip` → `paper_overleaf/` (working).

*All three are frozen. Nothing proposed.*

### 4.2 `venvs/` vs `.venv_rtable/` — **not duplicates; disjoint**

| Path | Size | pyvenv.cfg `command` | Purpose |
|---|---|---|---|
| `.venv_rtable/` | 1.5 GB | `…\.venv_rtable` | **RUNTIME.** Hosts `rapid_table` (SLANet-plus) for `rtable_child.py`. |
| `venvs/gpu/` | — | `…\.venv_gpu` ⚠ | CUDA ORT for `PRISM_ORT_GPU=1`. **Created as `.venv_gpu`, later renamed** — cfg still points at the old path. |
| `venvs/mineru_cpu/` | — | `…\venvs\mineru_cpu` | MinerU CPU baseline |
| `venvs/ppocr/` | — | `…\venvs\ppocr` | PP-StructureV3 baseline |
| `venvs/smol/` | — | `…\venvs\smol` | SmolDocling baseline |
| `venvs/docling_rebuttal/` | — | `…\venvs\docling_rebuttal` | Docling rebuttal experiment |

**`venvs/rtable` does not exist.** `rtable_worker.py:27` prefers it and falls back
to `.venv_rtable` — the fallback is what runs. README documents this accurately
("currently `.venv_rtable` at root until the in-flight benchmark run releases it").
The migration is **half-done**: the code is ready, the directory was never moved.

All six are Windows venvs (`Scripts/`, `python.exe`), all Python 3.12.6 from
`C:\Python312`, all with absolute paths baked into `pyvenv.cfg` and console
scripts. **None is relocatable or portable.** 11.5 GB total.

### 4.3 `preds/` vs `outputs/` vs `results/` vs `benchmarks/compare/preds_*` — **four distinct roles, no overlap**

| Path | Written by | Contains |
|---|---|---|
| `outputs/<stem>_output/` | `orchestrate.py:213`, `app.py:74` | **Product output** — `main.tex`, `main.pdf`, `assets/`, `logs/`. One dir per input image. 95 stale runs. |
| `preds/<run>/` | `benchmarks/*` | **PRISM benchmark predictions** — Markdown per page, for the eval harness. 113 named runs. 16.6 GB. |
| `benchmarks/compare/preds_*/` | `benchmarks/compare/run_*.py` | **External baseline predictions** — MinerU, Nougat, GOT2, olmOCR, Qwen2.5-VL, dots, PP-DocLayout. 7 dirs. |
| `results/rebuttal/` | `scripts/rebuttal/*` | **Rebuttal experiment JSON + LOG.md**, per `instructions.txt`. 2.7 GB. |

Genuinely different things that merely sound alike. The overlap risk is *naming*,
not content. Note the asymmetry: three are gitignored, `results/` is not.

### 4.4 `figures/` vs `paper/fig/` — **different kinds of thing, and `figures/` is unreferenced**

`paper/fig/` (5 files, identical in `paper_overleaf/fig/`): `arch.tex`,
`frontier.tex`, `splice.tex`, `teaser.tex`, `splice_input.png` — TikZ/pgfplots
sources `\input{}`-ed by `main.tex`.

`figures/` (12 entries, 18.1 MB): compiled PDFs — `ablation_waterfall.pdf`,
`camera_verification_grid.pdf`, `efficiency_frontier.pdf`, `failure_gallery.pdf`,
`formula_geometry.pdf`, `qualitative_io.pdf`, `thread_scaling.pdf`,
`normalisation_verified.{pdf,tex,_300dpi.png}`, `multilingual/`, `S2L (23).pdf`.

Grepping every `\includegraphics` and `\input` in `paper/`:
```
\includegraphics[width=8.2cm]{fig/splice_input}
\includegraphics[width=.33\linewidth]{example-image-golden}   ← template placeholder
\includegraphics[width=0.8\linewidth]{egfigure.eps}           ← template placeholder
\input{fig/arch} \input{fig/frontier} \input{fig/splice}
```

**No file in `figures/` is referenced by `paper/` or `paper_overleaf/`.** Two
`\includegraphics` calls still point at unfilled CVPR-template placeholders. Either
the `figures/` PDFs feed a different document (the `S2L (*).pdf` submission builds),
or figure inclusion is pending. **This needs your confirmation — I am not
proposing any action on `figures/`, which is frozen.**

### 4.5 `prism_supplementary.zip` vs `prism_supplementary_clean/`

- `prism_supplementary.zip` — 159.5 MB, **1511 entries**, includes `Texo/model/`, `models/ppdoclayoutv3/`, an embedded anonymised `.git/`, `EXPERIMENT_LOG.md`, `CHECKSUMS.txt`.
- `prism_supplementary_clean/prism-supplementary/` — 162.5 MB, **64 files**, same code tree (`pipeline/` 14, `normalization/` 6, `benchmarks/` 4, `configs/`, `models/ppdoclayoutv3`), but **no `.git/`, no `Texo/`, no `EXPERIMENT_LOG.md`**.

The `_clean` directory is a **later, reduced staging tree** — not an extraction of
the zip. `RELEASE_REPORT.md` (untracked, 2026-08-04) documents the zip at
"154.94 MB", which matches neither exactly; treat the report as describing an
earlier build. Both frozen; nothing proposed.

### 4.6 `paperresults.md` (root) vs `docs/paperresults.md` — **root is stale**

Different documents, not copies:
- root: 134 lines, *"SOTA Comparison on OmniDocBench **v1.5**"*, "11 systems self-run"
- `docs/`: 206 lines, *"Results vs. State of the Art"*, **v19 run (2026-07-07)**, full official dataset, cites `preds/odb_full_v19` and `omnidocbench_eval/result/odb_full_v19_*`

`README.md` links **only** `docs/paperresults.md`. The root copy is superseded.

### 4.7 `S2L (23).pdf` — exact duplicate

`S2L (23).pdf` and `figures/S2L (23).pdf` are byte-identical (md5 `c47723bda0e6f3ff1767397cdd01d021`, 2,800,798 B). Both untracked. `figures/` is frozen, so the root copy is the movable one.

---

## 5. PROPOSED TREE

### 5.1 ⚠ Precondition — do this before moving one byte of runtime code

The runtime derives its root as *"the parent of my package directory"* (§3.3, 6 sites).
Moving `pipeline/` into `src/` silently redefines that root as `src/`.

**Step 0 (code change, Phase 2):** add `src/prism/paths.py` with one explicit
resolver — anchored on a repo marker (`pyproject.toml`) or a `PRISM_ROOT`
environment override — and replace all 6 `ROOT_DIR`/`_ROOT` definitions with an
import from it. Only then perform the moves. Migrating in the other order
guarantees a broken tree at every intermediate commit.

Because of this coupling, `weights/`, `models/`, and `Texo/` **must move in the
same commit as the `paths.py` change**, never before.

### 5.2 Target layout

```
prism/
├── src/prism/          runtime code (importable package)
├── data/               datasets + test images
├── assets/             figures, images
├── paper/              LaTeX + paper artifacts        [FROZEN — unchanged]
├── eval/               benchmarks, harness, results
├── scripts/            dev tooling
├── weights/            all model weights
└── _quarantine/        suspect, awaiting your review
```

### 5.3 Path-by-path destinations

**Runtime → `src/`**

| From | To | Note |
|---|---|---|
| `pipeline/` (14 `.py`) | `src/prism/pipeline/` | after §5.1 |
| `normalization/` (6 `.py`) | `src/prism/normalization/` | after §5.1 |
| `app.py` | `src/prism/web/app.py` | update `ROOT`, `web/index.html`, orchestrate subprocess path |
| `web/index.html` | `src/prism/web/static/index.html` | update `app.py:126` |
| — | `src/prism/paths.py` | **new** — the single root resolver |
| — | `src/prism/__init__.py` | **new** |

**Weights → `weights/`** (consolidates the current `weights/` + `models/` + `Texo/model/` split)

| From | To | Note |
|---|---|---|
| `weights/*.onnx`, `en_dict.txt` | `weights/ocr/` | **keep LFS tracking** |
| `models/ppdoclayout/ppdoclayout_plus_l.onnx` | `weights/layout/ppdoclayout_plus_l.onnx` | update `models_interface.py:197`, `orchestrate.py:78`, `run_omnidocbench.py:57` |
| `models/ppdoclayoutv3/PP-DocLayoutV3.onnx` | `weights/layout/PP-DocLayoutV3.onnx` | update `models_interface.py:200` |
| `Texo/model/` | `weights/texo/` | update `math_worker_onnx.py:696`, `models_interface.py:184` |
| `Texo/src/`, `Texo/config/` | `src/vendor/texo/` | vendored lib; update `models_interface.py:69` |
| `Texo/{assets,data,outputs,scripts,TechnoSelection,demo.ipynb,web_demo.py,benchmark_texo.py,run_texo.py,main.py,uv.lock,pyproject.toml,pyrightconfig.json}` | `_quarantine/texo_upstream/` | 108 PNGs + 5 notebooks of upstream demo material; §6.6 |
| `models/doclayout_yolo_docstructbench_imgsz1024.onnx` | `_quarantine/weights/` | dead, §6.1 |
| `models/MFD/` | `_quarantine/weights/` | dead, §6.2 |
| `models/.cache/` | `_quarantine/` | HF metadata stubs |

**Data → `data/`**

| From | To |
|---|---|
| `data/omnidocbench/` | `data/benchmarks/omnidocbench/` |
| `data/omnidocbench_full/` | `data/benchmarks/omnidocbench_full/` |
| `data/olmocr_bench/` | `data/benchmarks/olmocr_bench/` |
| `data/fox/` | `data/benchmarks/fox/` |
| `data/test_sets/table_test/` | `data/test_sets/table_test/` (unchanged) |
| `test_images/real/` | `data/test_images/real/` — **tracked; keep tracked** (paper Sec 4.6 capture set) |
| `test_images/{synthetic,crops,rotation_benchmark}/` | `data/test_images/…` |

**Assets → `assets/`**

| From | To |
|---|---|
| `docs/normalisation_pipeline.png`, `docs/normalise.png` | `assets/docs/` |
| `figures/**` | **unchanged — FROZEN** |
| `paper/fig/**` | **unchanged — FROZEN** |
| `S2L (23).pdf` | `_quarantine/` (byte-identical to the frozen `figures/` copy, §4.7) |
| `S2L (24).pdf` | `assets/submissions/S2L_24.pdf` |

**Paper → `paper/` (FROZEN)**

`paper/`, `paper_overleaf/`, `paper_overleaf.zip`, `prism_supplementary.zip`,
`prism_supplementary_clean/` — **all unchanged.** Once the submission unfreezes,
`paper_overleaf/` + `.zip` should be regenerated from `paper/` rather than
maintained by hand (§4.1).

Movable paper-adjacent docs:

| From | To |
|---|---|
| `paper.md` | `paper/EXPERIMENT_LOG.md` (README link update) |
| `RELEASE_REPORT.md`, `ABLATION_PROVENANCE.md`, `findings.md`, `results.md` | `paper/provenance/` — **commit these first** |
| `paperresults.md` (root) | `_quarantine/` — superseded, §6.4 |

**Eval → `eval/`**

| From | To |
|---|---|
| `benchmarks/*.py` (7 tracked) | `eval/benchmarks/` |
| `benchmarks/__init__.py` | `eval/benchmarks/__init__.py` |
| `benchmarks/compare/*.py` (14) | `eval/compare/` — **fix the 14 hardcoded paths here (§3.1)** |
| `benchmarks/compare/{compare20_subset.json,compare50_images.txt,metrics_collected.json,RESULTS.md}` | `eval/compare/` |
| `benchmarks/compare/{preds_*,mineru_*,formula_eval}/` | `eval/_runs/compare/` (gitignored) |
| `omnidocbench_eval/` | `eval/omnidocbench_eval/` — **register as a real submodule (§0.3)** |
| `omnidocbench_eval/result/` | `eval/results/omnidocbench/` — **promote out of the submodule and TRACK (§0.4)** |
| `preds/` | `eval/_runs/preds/` (gitignored) |
| `outputs/` | `_runs/outputs/` (gitignored) |
| `results/rebuttal/` | `eval/results/rebuttal/` — **commit first (§0.5)** |
| `docs/section_scores_odb_full_v*.md` | `eval/results/section_scores/` |

**Scripts → `scripts/`**

| From | To |
|---|---|
| `scripts/{validate_ppdoclayout_onnx,export_texo_distill}.py` | `scripts/tools/` |
| `scripts/{bootstrap_margin,build_panel_mixed}.py` | `scripts/figures/` — **untracked; commit first** |
| `scripts/rebuttal/` (11) | `scripts/rebuttal/` — **untracked; commit first** |
| `scratchpad_runs/figures_wacv/{ablation_waterfall,efficiency_frontier}.py` | `scripts/figures/` — **§7.5, these generate frozen paper figures** |
| `scratchpad_runs/**` (rest) | `_quarantine/scratchpad_runs/` |

**Docs / config → root**

| From | To |
|---|---|
| `README.md` | unchanged — **fix the broken `docs/formula_fix_v2.md` link** |
| `docs/{context,paperresults,pipeline_audit_2026-07-04,weakness_analysis_v15}.md` | `docs/` (unchanged) |
| `pyproject.toml` | unchanged path — **contents need §7.2 correction** |
| `uv.lock` | regenerate after §7.2 |
| `.python-version` | **change `3.13` → `3.12`** to match every venv and CI reality |
| `.gitignore`, `.gitattributes` | update paths; **keep the LFS rules** |
| `STATUS.md`, `SESSION_HANDOFF.md`, `instructions.txt` | `docs/history/` |
| `skill.md` | `_quarantine/` — §6.3 |
| `.vscode/settings.json` | `_quarantine/` — §6.5 |

**Environments — leave in place, do not move**

`.venv_rtable/`, `venvs/*` have absolute paths baked into `pyvenv.cfg` and every
console script. Moving them breaks them silently. Replace with a documented
`scripts/setup_rtable_venv.{sh,ps1}` (§7.3) and delete the old trees only after
the recreated one is verified.

---

## 6. QUARANTINE LIST

Nothing here is a deletion proposal. Everything moves to `_quarantine/` for your review.

| # | Path | Size | Evidence |
|---|---|---|---|
| 6.1 | `models/doclayout_yolo_docstructbench_imgsz1024.onnx` | 75.3 MB | Zero matches for `doclayout_yolo` across all first-party `.py`. Superseded by PP-DocLayout (`run_omnidocbench.py:54`: *"replaces YOLOv11n + DocLayout + MFD"*). |
| 6.2 | `models/MFD/YOLO/yolo_v8_ft_640_dyn.onnx` | 174.5 MB | Zero matches for `yolo_v8_ft` or `MFD/` as a path. Same comment confirms replacement. |
| 6.3 | `skill.md` | 8.2 KB | Frontmatter `name: frontend-design`, `description: "Guidance for distinctive, intentional visual design when building new UI"`. An Anthropic agent-skill file. No PRISM content. Tracked — committed by accident. |
| 6.4 | `paperresults.md` (root) | 6.9 KB | Superseded by `docs/paperresults.md` (v19 vs v1.5). README links only the `docs/` copy. §4.6. |
| 6.5 | `.vscode/settings.json` | 1 line | Auto-approve rule for `.venv/Scripts/python test_rapid_ocr.py`; neither `.venv/` nor `test_rapid_ocr.py` exists. |
| 6.6 | `Texo/{assets,data,outputs,TechnoSelection,demo.ipynb,web_demo.py,benchmark_texo.py,run_texo.py,main.py}` | ~200 MB | Upstream demo material: 108 PNGs, 5 notebooks, a Gradio demo. Runtime touches only `Texo/model/` and `Texo/src/`. **Check the vendored LICENSE before removing anything permanently.** |
| 6.7 | `pipeline/__pycache__/tatr_worker_onnx.cpython-312.pyc` | — | Orphan `.pyc`; its `.py` is deleted-but-uncommitted. Zero `tatr` matches in live code. |
| 6.8 | `_web_uploads/fb2a1a08…png` | 0.4 MB | Orphaned upload from a crashed `app.py` job; `app.py:25` recreates the dir. |
| 6.9 | `.cache/huggingface/`, `models/.cache/`, `Texo/model/.cache/` | ~0 | HF download metadata stubs, no payload. |
| 6.10 | `S2L (23).pdf` (root copy only) | 2.8 MB | md5-identical to the frozen `figures/S2L (23).pdf`. **Quarantine the root copy; never touch the `figures/` one.** |
| 6.11 | `outputs/` (95 dirs) | 100 MB | Regenerable per-image product output. Gitignored. Ordinary run residue. |
| 6.12 | `scratchpad_runs/` (13 of 15 subdirs) | ~700 MB | Gitignored ad-hoc experiments. **Exclude `figures_wacv/` and anything else that generates a paper figure — see §7.5.** |
| 6.13 | `SESSION_HANDOFF.md` | 9.7 KB | Handoff for branch `wacv-results-hardening`, dated 2026-07-07; current branch is `latency-hardening`. Historical. |
| 6.14 | `pipeline/tatr_worker_onnx.py`, `scripts/export_tatr_onnx.py` | — | Deleted in the working tree, still tracked. Zero live `tatr` references. **Commit the deletion rather than restoring.** |

**Explicitly NOT quarantined despite looking dead:**
- `pipeline/rtable_child.py` — never imported; spawned as a subprocess (`rtable_worker.py:53`). A pure import scan calls this dead. It is not.
- `pipeline/formula_v2.py` — imported lazily at `page_core.py:914`.
- `benchmarks/compare/bench_metrics.py` — imported as a bare top-level name by 4 sibling scripts.
- `test_images/` — 293 tracked files, 311 MB, and the paper's Sec 4.6 capture set depends on it.

---

## 7. HANDOVER GAPS

What a new teammate **cannot** reconstruct from the repo alone.

### 7.1 🔴 The layout model is missing and undocumented

`models/` is gitignored. `models/ppdoclayout/ppdoclayout_plus_l.onnx` (129.7 MB) is
required — `orchestrate.py:102` raises `RuntimeError: PP-DocLayout model missing`
without it. **No download URL, no checksum, no fetch script exists anywhere in the
repo.** Same for `models/ppdoclayoutv3/PP-DocLayoutV3.onnx` and `Texo/model/`
(also gitignored, 76 MB of ONNX).

Needed: a `scripts/fetch_weights.py` with source URLs and SHA256s. Note
`prism_supplementary.zip` *does* bundle `Texo/model/` and `models/ppdoclayoutv3/`
plus a `CHECKSUMS.txt` — that zip is currently the only machine-readable record of
what the weights should be.

### 7.2 🔴 `pyproject.toml` misdeclares dependencies in both directions

Module-level (non-lazy) third-party imports across `pipeline/` + `normalization/`
are exactly: `numpy`, `PIL`, `cv2`, `onnxruntime`, `scipy`. Lazily imported:
`rapidocr_onnxruntime`, `tokenizers`, `psutil`.

| Declared as core | Reality |
|---|---|
| `torch>=2.10.0` | **Not needed.** Lazy-only, inside the Texo *torch fallback* in `models_interface.py`. `run_omnidocbench.py:30` states *"the main process is torch-free"*. |
| `torchvision>=0.25.0` | Not imported anywhere in first-party code. |
| `transformers>=4.57.6` | Lazy-only, same fallback path. |
| `ultralytics>=8.4.24` | **Not imported at all** — only named in comments. The YOLO detector it served is dead (§6.1). |
| `pdf2image>=1.17.0` | **Zero imports.** |

That is roughly 3 GB of install for a project whose headline claim is "283 MB of
weights, CPU-only".

| Needed, undeclared | Where |
|---|---|
| `rapid_table` | `rtable_child.py:35` — lives in the child venv only |
| `apted`, `lxml`, `Levenshtein` | TEDS scoring; README names them in the `.venv_rtable` recreate hint |
| `pyyaml`, `rapidfuzz` | in `benchmark` extra, but the harness needs them by default |
| `docling`, `paddleocr`, `paddle`, `surya`, `qwen_vl_utils` | `benchmarks/compare/*` baselines — each needs its own venv |

Also: `pyproject.toml` still cites "TATR" in a comment for code that no longer exists.

### 7.3 🔴 The table path depends on a second interpreter that must be built by hand

`PRISM_RTABLE` defaults to **on**. `rtable_worker.py:27-30` looks for
`venvs/rtable/Scripts/python.exe`, then `.venv_rtable/Scripts/python.exe`.
`venvs/rtable` does not exist; `.venv_rtable` (1.5 GB) is gitignored.

`available()` returns `False` when the interpreter is missing, so the pipeline
**silently degrades to the coordinate-heuristic table fallback** — no error, no
warning, and TEDS drops by roughly the +29 the docstring credits to SLANet-plus. A
teammate would see materially worse table numbers and have no idea why.

The only recreate instruction is a parenthetical in `.gitignore`:
`pip install rapid-table rapidocr apted lxml Levenshtein`. It should be a script.

**Both candidate paths are Windows-only** (`Scripts/python.exe`). On Linux/macOS
the table path can never activate.

### 7.4 🟠 Five benchmark venvs with no build recipe

`venvs/{gpu,mineru_cpu,ppocr,smol,docling_rebuttal}` — 10.0 GB, gitignored, no
requirements files. `benchmarks/compare/run_mineru.py:14` additionally points at
`.venv_mineru\Scripts\mineru.exe`, **which does not exist** (the venv is
`venvs/mineru_cpu`) — that runner is already broken on this machine.

Pinned versions for the MinerU CPU baseline exist only in
`scratchpad_runs/mineru_cpu/STATE.md`, inside a gitignored directory.

### 7.5 🟠 The paper's figure generators are gitignored

`figures/ablation_waterfall.pdf` and `figures/efficiency_frontier.pdf` are tracked
paper figures. Their generators —
`scratchpad_runs/figures_wacv/{ablation_waterfall,efficiency_frontier}.py` —
are inside gitignored `scratchpad_runs/`. `scripts/build_panel_mixed.py`
(generator for `figures/multilingual/`) is untracked.

**Two tracked paper figures cannot be regenerated from a fresh clone.** Rescue
these before any cleanup touches `scratchpad_runs/`.

### 7.6 🟠 ~50 `PRISM_*` environment variables, roughly half undocumented

50 distinct `PRISM_*` vars are read across `pipeline/`, `normalization/` and
`benchmarks/`. 24 appear in `README`/`STATUS`/`docs`. The remaining **26 are
documented nowhere**:

```
PRISM_AFFINITY  PRISM_BAND_CLASSES  PRISM_BAND_DROP  PRISM_BAND_GAP
PRISM_BAND_HCAP  PRISM_DET_LIMIT  PRISM_DUMP_BLOCKS  PRISM_EVAL_MATCH_WORKERS
PRISM_EVAL_TEDS_WORKERS  PRISM_FML_TIMEOUT_BASE  PRISM_FML_TIMEOUT_PER
PRISM_INLINE_FML  PRISM_LAYOUT_CACHE  PRISM_MAX_SHORTER  PRISM_MOIRE
PRISM_OCR_MAX_SIDE  PRISM_OCR_MAX_SIDE_SS  PRISM_PPDL_BASE_CONF
PRISM_PROBE_LOG  PRISM_PROBE_MIN_SIDE  PRISM_RES_FLOOR  PRISM_RO_V2
PRISM_SINGLE_WORKER  PRISM_STAGE_TIMING  PRISM_STITCH_CHUNK  PRISM_TABLE_GEOM
PRISM_VISUAL_FIDELITY
```

Several change measured results, so this is a reproducibility gap, not just a
docs gap. Two carry non-obvious semantics worth writing down explicitly:
- `PRISM_VISUAL_FIDELITY=1` — set by `app.py:62` only. The **web UI runs a
  different layout mode than the benchmark**, contradicting the README's "serves
  the exact benchmark build".
- `PRISM_NORM_STRICT=1` — `setdefault` at `run_omnidocbench.py:26`. Benchmark runs
  suppress the product-mode shadow rescue. **Benchmark and product behaviour differ
  by default**; a teammate reproducing a score outside the harness gets different
  output.

### 7.7 🟠 System tools assumed present

| Tool | Where | Consequence if absent |
|---|---|---|
| `xelatex` / `pdflatex` | `app.py:87` | Web UI returns "PDF compilation failed"; `orchestrate.py` still emits `.tex` |
| `pdftoppm` (poppler) | `run_olmocr_bench.py:84` | olmOCR benchmark cannot rasterise |
| **git-lfs** | clone time | `weights/*.onnx` arrive as 130-byte pointers → opaque ONNX parse failure (§0.1) |
| TeX Live 2026 | CDM formula scoring | `docs/paperresults.md` pins this version for reproducing formula numbers |

None is mentioned in `pyproject.toml`. README mentions only pdflatex/xelatex.

### 7.8 🟠 Platform assumptions

- `.venv_rtable/Scripts/python.exe` — Windows-only (§7.3). No POSIX branch.
- `run_mineru.py:14` — `.exe`
- Every `pyvenv.cfg` — `home = C:\Python312`
- `run_omnidocbench.py:33` reconfigures stdout to UTF-8 to survive Windows `cp1252`
- `orchestrate.py:21` uses `psutil.cpu_affinity()` — unsupported on macOS
- 29 hardcoded `C:\PROJECTS\s2l2\testprism` literals (§3.1)
- `instructions.txt` states the reference machine: *"Windows 11, 16 logical cores, CPU only"*; paper latency tables are i7-11800H-specific

**Nothing in the repo has ever run on Linux or macOS.** If the teammates are not on
Windows, that is the largest single item of work and it is not visible from the README.

### 7.9 🟡 Phantom and broken references

| Reference | Status |
|---|---|
| `orchestrate.py:69` → `evaluation.profiler` | Module does not exist. Wrapped in `try/except ImportError`, so `--profile` **silently does nothing**. |
| `models_interface.py:70` → `ROOT_DIR/text-table-latex` | Directory does not exist; harmless `sys.path.append`. |
| `README.md` → `docs/formula_fix_v2.md` | File does not exist. |
| `run_mineru.py:14` → `.venv_mineru` | Does not exist (§7.4). |
| `rtable_worker.py:27` → `venvs/rtable` | Does not exist; falls back (§4.2). |
| `.vscode/settings.json` → `test_rapid_ocr.py` | Does not exist (§6.5). |

### 7.10 🟡 Version and provenance inconsistencies

- `.python-version` says **3.13**; `pyproject` says `>=3.12`; all six venvs are **3.12.6**.
- `README.md` describes `venvs/rtable` as the current home — it does not exist yet (README is honest about this, but it reads as done).
- `docs/section_scores_odb_full_v{14,16,17,20}.md` are tracked; the current build is **v21** and `README`/`docs/paperresults.md` quote **v19/v20** numbers. Per your memory note, v21 has a known table-duplication artifact — **not visible anywhere in the tracked docs.** A teammate reading `README.md` would take v21 output at face value.

---

## 8. Suggested Phase-2 order

Sequenced so the tree is never broken mid-migration.

1. **Back up `omnidocbench_eval/result/` off-machine** (§0.4). Nothing else until this is done.
2. **Commit or explicitly quarantine all uncommitted work** (§0.5): `results/`, `scripts/rebuttal/`, the four untracked reports, the two pending TATR deletions.
3. **Rescue the gitignored figure generators** (§7.5) into `scripts/figures/`.
4. **Add `.gitmodules`** for `omnidocbench_eval` (§0.3); promote `result/` out of the submodule and track it.
5. **Introduce `src/prism/paths.py`** and switch all 6 root resolvers to it (§5.1). *No moves yet — verify the pipeline still runs.*
6. **Move runtime + weights in one commit** (§5.3), re-run a single-image smoke test and one benchmark page.
7. **Fix the 29 hardcoded paths** in `eval/compare/` and `scripts/` (§3.1), and the eval-config emitter (§3.2).
8. **Correct `pyproject.toml`** (§7.2); regenerate `uv.lock`; set `.python-version` to `3.12`.
9. **Write `scripts/fetch_weights.py`** (§7.1) and `scripts/setup_rtable_venv.*` (§7.3), with a POSIX branch.
10. **Document the 26 undocumented env vars** (§7.6), flagging `PRISM_VISUAL_FIDELITY` and `PRISM_NORM_STRICT` as behaviour-divergent.
11. **Move quarantine candidates** into `_quarantine/` (§6) — your review, no deletions.
12. **Unfreeze check:** once the submission clears, reconcile `paper_overleaf/` from `paper/` (§4.1) and resolve the `figures/` reference question (§4.4).

---

## Appendix — measured sizes

| Path | Size | Files | Tracked |
|---|---|---|---|
| `preds/` | 16,586.8 MB | 79,400 | 0 |
| `venvs/` | 9,990.2 MB | 214,009 | 0 |
| `data/` | 5,463.3 MB | 19,173 | 0 |
| `results/` | 2,700.6 MB | 659 | **0 (and not ignored)** |
| `.venv_rtable/` | 1,501.5 MB | 25,862 | 0 |
| `omnidocbench_eval/` | 1,425.4 MB | 2,467 | gitlink only |
| `scratchpad_runs/` | 737.5 MB | 1,182 | 0 |
| `models/` | 510.1 MB | 6 | 0 |
| `benchmarks/` | 405.3 MB | 1,486 | 37 |
| `test_images/` | 310.7 MB | 293 | **293** |
| `Texo/` | 295.8 MB | 260 | 222 |
| `prism_supplementary_clean/` | 162.5 MB | 64 | 0 |
| `outputs/` | 100.2 MB | 607 | 0 |
| `weights/` | 38.8 MB | 4 | **4 (3 via LFS)** |
| `figures/` | 18.1 MB | 28 | 5 |
| `docs/` | 3.7 MB | 10 | 10 |
| `paper/` | 0.7 MB | 28 | 19 |
| `pipeline/` | 0.6 MB | 30 | 16 |
| `paper_overleaf/` | 0.2 MB | 19 | 19 |
| `normalization/` | 0.1 MB | 12 | 6 |
| `scripts/` | 0.1 MB | 15 | **3** |
| `web/` | ~0 MB | 1 | 1 |
| **Working tree total** | **≈ 40 GB** | ≈ 346,000 | — |
