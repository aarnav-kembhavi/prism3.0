# PRISM — setup

Getting PRISM running on a machine that has never run it.

> **This has only ever run on Windows.** Windows 11, Python 3.12, 16 logical
> cores, CPU only. Nothing here has been executed on Linux or macOS. Some paths
> are Windows-shaped by construction (the table worker looks for
> `Scripts\python.exe` first). POSIX fallbacks exist in a few places but are
> **untested** — treat a non-Windows setup as porting work, not installation.

---

## 0. Before you clone: install Git LFS

Three OCR models are stored in Git LFS. If you clone without LFS you get
130-byte text pointers where the models should be, and the pipeline dies at
model load with an opaque ONNX Runtime parse error that does not mention LFS.

```bash
git lfs install          # once per machine, BEFORE cloning
git clone --recurse-submodules <repo-url>
cd testprism
```

Already cloned without it?

```bash
git lfs install
git lfs pull
```

Verify — each file must be megabytes, not bytes:

```powershell
Get-ChildItem weights\*.onnx | Select-Object Name, Length
```

Expected: `PP-OCRv6_det_small.onnx` 9,929,594 · `PP-OCRv6_rec_small.onnx`
21,234,383 · `en_PP-OCRv4_rec.onnx` 7,652,836.

---

## 1. Python environment

Python **3.12** (`.python-version` pins 3.12; every venv on the original machine
is 3.12.6). 3.13 is untested.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

That installs only what `pipeline/` and `normalization/` actually import:
numpy, onnxruntime, opencv-python, pillow, psutil, scipy,
rapidocr-onnxruntime, tokenizers, pyyaml, rapidfuzz.

Optional extras:

| Extra | Install | What it adds |
|---|---|---|
| `benchmark` | `pip install -e ".[benchmark]"` | OmniDocBench / Fox / olmOCR scoring |
| `torch-fallback` | `pip install -e ".[torch-fallback]"` | torch + transformers Texo fallback. **Not needed** — the default path is ONNX-only and torch-free |
| `export` | `pip install -e ".[export]"` | rebuilding the Texo ONNX graphs (section 3) |
| `rtable` | *do not install into this env* | records the child-venv versions — see section 4 |

`uv.lock` is currently **stale** against `pyproject.toml`. Use `pip` as above, or
regenerate with `uv lock` first.

---

## 2. System tools

Not installable via pip, not checked at startup — you find out when something
fails.

| Tool | Needed for | Without it |
|---|---|---|
| **git-lfs** | cloning (section 0) | OCR models arrive as text pointers |
| **xelatex** / **pdflatex** | `app.py` PDF compilation | `.tex` is still written; PDF step reports "PDF compilation failed". `xelatex` is used when the document contains `\usepackage{xeCJK}` (CJK pages), `pdflatex` otherwise — install a TeX distribution with both |
| **pdftoppm** (poppler) | `benchmarks/run_olmocr_bench.py` only | that benchmark cannot rasterise its PDFs |
| **TeX Live 2026** | CDM formula scoring | formula numbers will not reproduce; `docs/paperresults.md` pins this version |

`pipeline/orchestrate.py` itself needs **none** of these — it writes `.tex` and
Markdown without touching LaTeX.

---

## 3. Model weights — **not in the repo**

`models/` is gitignored, so a fresh clone has **no layout detector and no
formula model**, and the pipeline cannot start. There is no fetch script; these
files have to be brought over from a machine that already has them, or rebuilt.

### Required by the default path

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `models/ppdoclayoutv3/PP-DocLayoutV3.onnx` | 130,502,049 | `d24809294b2f9f1a9a2767043a64df2714b66e5be056887be2233d1117d784f6` |
| `Texo/model/onnx/encoder_model.onnx` | 54,168,538 | `d6162b4e0d2a2fba124480bc487a9b6c2bd01636d0b61a1f6833e056a7263c1a` |
| `Texo/model/onnx/decoder_model_merged.onnx` | 27,718,980 | `7ea630c6eb84e28b575b09c8be0f03cdd2b4f21b6717ddd51b5e82a888631053` |
| `Texo/model/tokenizer.json` | 27,084 | `908bf661a92ca6c6d92817877bd819231438de94f06ab8683dc9b0a2ef7a8297` |

> **Watch out:** if the layout model is missing, `orchestrate.py` raises
> `PP-DocLayout model missing: ...ppdoclayout_plus_l.onnx`. **That message names
> the wrong file.** `PRISM_PPDL_V3` defaults to `1`, so the file actually being
> loaded is `models/ppdoclayoutv3/PP-DocLayoutV3.onnx`. Only if you set
> `PRISM_PPDL_V3=0` does it want `ppdoclayout_plus_l.onnx`
> (129,736,329 bytes, `d86d50c5040702f42326885644805600d5822a97c694fb5ea3a7a876df050ee1`).

Arriving via Git LFS, already present after section 0:

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `weights/PP-OCRv6_det_small.onnx` | 9,929,594 | `090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f` |
| `weights/PP-OCRv6_rec_small.onnx` | 21,234,383 | `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884` |
| `weights/en_PP-OCRv4_rec.onnx` | 7,652,836 | `02672d591fee561a95a2491bd5504d66c812b145d1043026c5658ecc165c7e85` |
| `weights/en_dict.txt` | 190 | `5662df9d2d03f0e8ca0d3b0649d6acbab904b6a14b3d3521463c71c37c668ce3` |

Verify anything:

```powershell
Get-FileHash -Algorithm SHA256 models\ppdoclayoutv3\PP-DocLayoutV3.onnx
```

### Rebuilding the two Texo graphs

They are exported from a public distilled checkpoint rather than shipped:

```powershell
pip install -e ".[export]"
python scripts/export_texo_distill.py
```

ONNX export is **not byte-reproducible** — a rebuilt graph is functionally
equivalent but has a different SHA-256. The hashes above identify the exact
graphs behind the reported numbers; they are provenance, not a verification
target. A mismatch after rebuilding is expected and is not a failure.

### Not required

`models/doclayout_yolo_docstructbench_imgsz1024.onnx` (75 MB) and
`models/MFD/` (174 MB) are dead — no code references them. Do not copy them.

---

## 4. RapidTable child venv

Table structure recognition (SLANet-plus, +29 TEDS) runs in a **separate
interpreter**, because `rapid_table` pins a `rapidocr` major version that
conflicts with the main environment. It is spawned as a child process.

```powershell
python -m venv venvs\rtable
venvs\rtable\Scripts\python -m pip install rapid_table rapidocr
```

Build it at `venvs\rtable` — that is the path the worker prefers. `.venv_rtable`
at the repo root is accepted as a legacy fallback.

The SLANet-plus graph itself is **not** downloaded by you: `rapid_table` fetches
it into its own site-packages on first use (~7.8 MB).

Versions known to work: `rapid-table` 3.0.2, `rapidocr` 3.9.1, on Python 3.12.6.

**If you skip this step the pipeline now stops with an error** naming every path
it tried and the command above. That is deliberate — it used to fall back
silently to a coordinate heuristic, which produced quietly-much-worse tables
with no indication why. On a two-table test page the fallback produced **zero**
tables.

To run deliberately without it:

```powershell
$env:PRISM_RTABLE = "0"
```

---

## 5. Benchmark harness (optional)

`omnidocbench_eval/` is a submodule. If you did not clone with
`--recurse-submodules`:

```bash
git submodule update --init
```

Then apply the UTF-8 patch — without it the harness raises `UnicodeDecodeError`
on CJK pages under Windows, because it opens `.md`/`.json` with the cp1252
default:

```bash
cd omnidocbench_eval
git apply ../patches/omnidocbench-utf8.patch
cd ..
```

The submodule is pinned to upstream `2b161d0`; the patch adds
`encoding='utf-8'` in `src/core/pipeline_eval.py` and
`src/dataset/end2end_dataset.py`.

---

## 6. Datasets — **not in the repo**

`data/` is gitignored (5.5 GB). A fresh clone has none of it, and every
benchmark entry point below needs it.

| Path | What | Source |
|---|---|---|
| `data/omnidocbench/`, `data/omnidocbench_full/` | OmniDocBench images + GT | [opendatalab/OmniDocBench](https://github.com/opendatalab/OmniDocBench) |
| `data/olmocr_bench/` | olmOCR-Bench | allenai/olmOCR-bench |
| `data/fox/` | Fox benchmark | Fox benchmark release |
| `data/test_sets/table_test/` | small table set | local |

`test_images/` **is** tracked (293 files) — so the single-image entry point in
section 7 works from a clean clone as soon as the weights are in place.

---

## 7. Running it

Every command below is real and runnable once sections 0–4 are done.

### Single image → LaTeX (the main entry point)

```powershell
python pipeline/orchestrate.py test_images/real/clean/ieee_p6_twocol_tables.png
```

Writes `outputs/ieee_p6_twocol_tables_output/main.tex` plus `assets/` and
`logs/`. Takes ~20 s cold on an i7-11800H (model load dominates; steady-state is
~5.9 s/page). This exact command is the smoke test — if it completes and
`main.tex` contains `<table>` rows with real cell text, the whole stack is
working.

### Web UI

```powershell
python app.py
```

Serves `http://localhost:8000`. Upload an image, get a PDF. Needs
xelatex/pdflatex (section 2). Note it sets `PRISM_VISUAL_FIDELITY=1`, so the UI
reproduces the page's column layout rather than the flat benchmark stream —
**the UI output is not identical to the benchmark build.**

### OmniDocBench

```powershell
# 18-page bundled demo, no dataset download needed
python benchmarks/run_omnidocbench.py

# full run against a real dataset
python benchmarks/run_omnidocbench.py `
    --gt-json data/omnidocbench/OmniDocBench_available.json `
    --images-dir data/omnidocbench/images `
    --pred-dir preds/my_run

# re-score existing predictions without re-running the pipeline
python benchmarks/run_omnidocbench.py --pred-dir preds/my_run --eval-only
```

### Other benchmarks

```powershell
python benchmarks/run_fox.py --pred-dir preds/fox
python benchmarks/run_olmocr_bench.py          # needs pdftoppm
python benchmarks/benchmark_glare.py --skip-run
```

### Figures

```powershell
python scripts/figures/ablation_waterfall.py
python scripts/figures/efficiency_frontier.py
```

Self-contained — data is inlined, no inputs needed. **They overwrite
`figures/*.pdf`, which are live conference submission artifacts.** Do not run
them casually.

### Rebuttal experiments

```powershell
python scripts/rebuttal/phase3_acceptance.py
```

Reads `test_images/real/defects/defects-images/`, writes `results/rebuttal/`
(gitignored).

---

## 8. Environment variables

~50 `PRISM_*` variables are read across the codebase. The ones that change
behaviour enough to surprise you:

| Variable | Default | Effect |
|---|---|---|
| `PRISM_RTABLE` | `1` | `0` disables table structure recognition. Missing venv is now a hard error (section 4) |
| `PRISM_PPDL_V3` | `1` | `0` switches to `ppdoclayout_plus_l.onnx` — a **different weight file** |
| `PRISM_VISUAL_FIDELITY` | unset | `1` reproduces column layout. `app.py` sets this; the benchmark does not |
| `PRISM_NORM_STRICT` | unset (benchmark sets `1`) | Suppresses the product-mode shadow rescue. **Benchmark and product behaviour differ by default** — reproducing a score outside the harness needs this set |
| `PRISM_ORT_GPU` | `0` | Optional CUDA runtime; needs a separate GPU venv |
| `PRISM_ONNX_THREADS` | auto | Caps ONNX/OpenMP threads |
| `PYTHONIOENCODING` | — | Set to `utf-8` if you write your own driver; entry points already reconfigure stdout |

`docs/context.md` covers the rest. Roughly 26 of them are undocumented — grep
`os.environ.get` in `pipeline/` if you hit something unexplained.

---

## 9. Known rough edges

- **`--profile` silently does nothing.** `orchestrate.py` imports
  `evaluation.profiler`, which does not exist in the repo; the import is wrapped
  in `try/except ImportError`.
- **`git status` reports `omnidocbench_eval` as modified** on the original
  machine. Expected — the local submodule checkout is deliberately ahead of the
  pinned upstream commit. Do not `git add omnidocbench_eval` to silence it; that
  restores an unfetchable SHA.
- **`uv.lock` is stale** against `pyproject.toml` (section 1).
- **`README.md` links `docs/formula_fix_v2.md`**, which does not exist.
- **Tracked per-section score docs stop at v20** while the current build is v21.
  Do not assume `docs/section_scores_odb_full_v20.md` describes what you just
  ran.
- **`benchmarks/compare/run_*.py` each need their own venv** (MinerU,
  PP-StructureV3, SmolDocling, Docling, GPU VLMs). Only `venvs/mineru_cpu` has a
  recorded recipe, and it lives in gitignored
  `scratchpad_runs/mineru_cpu/STATE.md`.

---

## 10. Quick check

```powershell
git lfs ls-files                                    # 3 files, not empty
python -c "import onnxruntime, cv2, rapidocr_onnxruntime; print('deps ok')"
python -c "from pipeline import rtable_worker as r; print('rtable:', r.available())"
python pipeline/orchestrate.py test_images/real/clean/ieee_p6_twocol_tables.png
```

Fourth command writing `outputs/ieee_p6_twocol_tables_output/main.tex` means
you are set up.
