# PRISM on Cloud Run

Container: `Dockerfile` · Server: `serve.py` · Region: `asia-south1`

## Prerequisites

The image bakes the **fp16** graphs, which are gitignored and rebuilt from the
fp32 originals. Do this once in a fresh checkout, or the build fails at COPY:

```bash
python quant/fp16_convert.py          # writes the *_fp16.onnx siblings
```

One-time GCP setup:

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com
gcloud artifacts repositories create prism \
    --repository-format=docker --location=asia-south1
```

## Build

```bash
PROJECT=$(gcloud config get-value project)
IMG="asia-south1-docker.pkg.dev/$PROJECT/prism/prism:v1"

gcloud builds submit --tag "$IMG" --timeout=2400s --region=asia-south1
```

## Deploy

Every flag below is non-default and load-bearing:

```bash
gcloud run deploy prism \
    --image "$IMG" \
    --region asia-south1 \
    --memory 4Gi \
    --cpu 4 \
    --concurrency 4 \
    --timeout 900 \
    --min-instances 0 \
    --max-instances 1 \
    --cpu-boost \
    --allow-unauthenticated \
    --session-affinity
```

| Flag | Why |
|---|---|
| `--memory 4Gi` | Startup peaks well above 2 GiB once the fp32 graphs are materialised; 2 GiB OOMs |
| `--cpu 4` | The pipeline is CPU-bound end to end |
| `--concurrency 4` | Page loads, `/health` and progress polls must not queue behind a 17 s parse. Memory is protected by an `asyncio.Semaphore(1)` around the parse itself, not by this number — see Concurrency below |
| `--timeout 900` | A dense multi-page PDF takes minutes |
| `--min-instances 0` | Scale to zero; cold start is the trade-off |
| `--max-instances 1` | **Not a cost knob.** `app.py`'s job registry is a per-instance dict; with more than one instance a `/status/{id}` poll can land on an instance that never saw the upload and the job is lost — see Concurrency below |
| `--cpu-boost` | Model load is CPU-bound, and it all happens before the first request |
| `--allow-unauthenticated` | Public endpoint |
| `--session-affinity` | Best-effort help for the same problem; not sufficient alone, which is why max-instances is pinned |

## Use

```bash
URL=$(gcloud run services describe prism --region asia-south1 \
      --format='value(status.url)')

curl -s "$URL/health"                                    # 200 only once loaded

# The UI's own chain (images only -- app.py's ALLOWED_SUFFIXES has no .pdf)
ID=$(curl -s -F "file=@page.png" "$URL/upload" | python -c "import sys,json;print(json.load(sys.stdin)['job_id'])")
curl -s "$URL/status/$ID"                                # {status, message, queue_position}
curl -s "$URL/pdf/$ID"   -o out.pdf                      # the right pane
curl -s "$URL/latex/$ID" -o out.tex                      # the left pane

# The API, which is what takes PDFs
curl -s -F "file=@page.png" "$URL/parse"                 # -> markdown
curl -s -F "file=@doc.pdf"  "$URL/parse"                 # -> markdown, per page
```

`/health` returns the startup timings and peak RSS it measured:

```json
{"status":"ok","back_conversion_s":…,"back_conversion_peak_rss_mb":…,
 "materialized_mb":…,"warmup_s":…,"startup_peak_rss_mb":…,"total_startup_s":…}
```

Uploads over **30 MB** are refused with 413 — Cloud Run caps request bodies at
32 MB, and a silent truncation is worse than a refusal.

## How the fp16 graphs are used

The image ships only `*_fp16.onnx` (126 MB rather than 251 MB). The pipeline
runs its models in **worker subprocesses**, so an in-process back-conversion in
the server would not reach them — each worker would redo the conversion on
every request. Instead `serve.py` materialises real fp32 graphs into
`PRISM_FP32_CACHE` (default `/tmp/prism_fp32`) **once at startup**, before the
port is served, and points the pipeline at those files through the per-graph
`PRISM_FP16_<KEY>` override. Workers then open ordinary fp32 graphs.

`/tmp` on Cloud Run is a tmpfs, so the materialised graphs count against the
memory limit. That is the main reason for `--memory 4Gi`.

slanet-plus is deliberately left fp32: the fp16 sweep measured mean similarity
0.937 for it against 0.996+ for every other graph (see `quant/README.md`).

## The UI

`GET /` serves **`web/index.html` exactly as committed at `2e62b83`**
(2026-06-29), through **`app.py` exactly as committed at `d91c6b1`**
(2026-07-14). Neither file is modified — the blobs in the image are
`7a81995` and `97b3b63`, the same git objects those commits hold. `serve.py`
imports `app.py` and splices its routes in; it registers no `/` of its own:

| route | from |
|---|---|
| `GET /` · `POST /upload` · `GET /status/{id}` · `GET /pdf/{id}` · `GET /latex/{id}` | `app.py`, verbatim |
| `POST /parse` · `GET /health` · `GET /progress` | `serve.py` |

Only `app.py`'s own routes are spliced, not `legacy.app.router` wholesale:
that router also carries the `/docs` and `/redoc` routes FastAPI attaches to
every app, which this service disables deliberately.

The page is a **LaTeX/PDF viewer**: upload an image, and the two panes are the
generated `main.tex` (highlight.js) and the compiled PDF (`<iframe>`). It
accepts **images only** — `app.py`'s `ALLOWED_SUFFIXES` has no `.pdf`. That is
the original's behaviour and is left alone; `POST /parse` is the route that
takes PDFs, and it returns markdown rather than driving the UI.

Everything the container needed for the original to run unchanged was done
outside those two files:

| what | where | why |
|---|---|---|
| `/app/_web_uploads` → `/tmp/prism_uploads` | Dockerfile symlink | `app.py:25` writes uploads beside itself; `/tmp` is the only tmpfs |
| `/app/outputs` → `/tmp/prism_outputs` | Dockerfile symlink | `orchestrate.py:213` hardcodes `<repo>/outputs` |
| those two `/tmp` targets created at import | `serve.py:_configure_env` | a *dangling* symlink is not a directory, so `UPLOAD_DIR.mkdir(exist_ok=True)` would raise |
| `port=8000` in `app.py`'s `__main__` | — | dead code under `serve.py`; `$PORT` is untouched |

`app.py` runs the pipeline as an `orchestrate.py` **subprocess per job**, which
inherits `os.environ` and therefore the `PRISM_FP16_<KEY>` overrides
`materialize_fp32_graphs()` sets at startup. That is why it works here
unchanged — but it also means a job reloads every model, so a `/upload` job is
slower than a `/parse` (which uses the persistent workers). That is the
original's design, not a regression introduced here.

`app.py` serialises its jobs with `_worker_busy`; `serve.py` serialises
`/parse` with a semaphore. The two knew nothing about each other, so a
`/upload` job and a `/parse` could have run two pipelines at once and exceeded
4Gi. `serve.py` now claims `app.py`'s flag using `app.py`'s own protocol —
lock, test, set, `_pump_queue()` on release — rather than editing `app.py`.

The in-build smoke test asserts `GET /` contains the original's own ids and
routes (`#latex-pre`, `#pdf-viewer`, `#drop-zone`, `fetch('/upload'`,
`/status/${jobId}`) plus one `.split` with exactly two `.pane` children, then
drives `POST /upload` → `GET /status/{id}` → `GET /pdf/{id}` and requires a
real PDF with a non-zero page count. Serving anything but the original page,
or shipping a TeX install that cannot compile, fails the build.

## TeX

The right pane is a compiled PDF, so the image needs a TeX. Debian's route to
`paracol` is `texlive-latex-extra`, about 1.6 GB for one `.sty`, so this uses
**TinyTeX**: `install-bin-unix.sh` lays down the binaries plus `tlmgr` and
nothing else, and every package is then named explicitly —

```
latex latex-bin latex-fonts pdftex xetex fontspec lm amsmath amsfonts
geometry graphics graphics-def graphics-cfg booktabs ms paracol xecjk
tools etoolbox xkeyval iftex unicode-data ec l3kernel l3packages l3backend
```

`ms` is `ragged2e` (and `everysel`, which `paracol` wants); `lm` is
`fontspec`'s default font under xelatex; `graphics*` is `graphicx`. TinyTeX
disables docfiles and srcfiles, which is most of what texlive weighs.

Fonts: **`fonts-noto-cjk`, not `-extra`**, and only the **Sans** family is kept
— one `.ttc` covers SC/TC/JP/KR — with `NotoSerifCJK*.ttc` deleted, since the
generated preamble asks for a single CJK family.

### The CJK font, and why `xelatex` is a wrapper

`pipeline/latex_builder.py` writes `\usepackage{xeCJK}` and **never calls
`\setCJKmainfont`**. With no CJK font configured xeCJK falls back to the Latin
font and xelatex then **exits 0 having dropped every ideograph** — a PDF full
of holes, which no exit-code check would catch. The preamble is pipeline code,
not deployment code, so the default is supplied from outside:
`deploy/xelatex-cjk.sh` is installed at `/usr/local/bin/xelatex`, earlier on
`PATH` than `/opt/texbin`, and re-invokes the real binary through the LaTeX
kernel's own package hook:

```
xelatex -jobname=main '\AddToHook{package/xeCJK/after}{\setCJKmainfont{Noto Sans CJK SC}}\input{main.tex}'
```

`-jobname` is load-bearing: `\input{...}` makes the jobname `texput`, so
`main.tex` would compile to `texput.pdf` and `app.py` would report "PDF
compilation failed" on a PDF that built correctly. `pdflatex` needs no wrapper
— `app.py:85` only picks xelatex when the document loads xeCJK.

### The build proves it compiles

`deploy/texcheck.py` compiles **real pipeline output**, not minimal examples,
using the same compiler-selection rule as `app.py:85`:

| fixture | preamble | engine |
|---|---|---|
| `deploy/texcheck/latin` | `inputenc` + `paracol` | pdflatex |
| `deploy/texcheck/cjk` | `xeCJK` + `paracol` | xelatex |

`paracol` is in both because `PRISM_VISUAL_FIDELITY=1` — which `app.py:64` sets
for **every** web job — is what makes a multi-column page emit it. The check
fails the build on a non-zero exit, a missing PDF, zero pages, any
`Missing character` / `No CJK font family` / `LaTeX Error` in the log, **and**
on fewer than 100 CJK glyphs read back out of the compiled PDF's text layer.
The last one is the one that matters: it is the only assertion that catches a
compile that succeeded and produced holes.

## Concurrency

`--concurrency 4`, with an `asyncio.Semaphore(1)` around the parse only.

A parse holds ~1.8 GB resident and the materialised fp32 graphs take another
251 MB of tmpfs against the same limit, so two concurrent pipelines would
exceed 4Gi. `--concurrency 1` enforces that but is the wrong tool once a UI
exists: the page, `/health` and every `/status/{id}` poll would queue behind a
17-second job and the app would look hung. Measured on the live service, with a
job running: **389 ms median** across `/health` and `/progress`.

Two independent serialisers had to be joined up. `app.py` runs one job at a
time behind `_worker_busy`; `serve.py` runs one `/parse` at a time behind its
semaphore. Neither knew about the other, so a `/upload` job and a `/parse`
could have run two pipelines at once. `serve.py` now claims `app.py`'s flag
using `app.py`'s own protocol rather than editing `app.py`.

**`--max-instances 1`, and this is not a cost decision.** `app.py`'s `_jobs`
dict is per-instance process state, and the original UI's whole flow depends on
it: `POST /upload` returns a `job_id`, then the page polls `GET /status/{id}`
every 2 s and finally fetches `GET /pdf/{id}` and `GET /latex/{id}`. Every one
of those lands on whichever instance Cloud Run picks. A poll that reaches a
different instance than the upload gets `404 Job not found`, which the page
surfaces as `Status check failed` and the job is lost — even though it is still
running, correctly, on the other instance.

`--session-affinity` is not sufficient on its own: it is best-effort and,
measured at `--max-instances 3`, did **not** hold while instances were being
created — three cold starts inside 70 s, and requests scattered across them. It
is kept anyway, since it costs nothing and helps when more than one instance
does exist.

Extra instances buy nothing here regardless: both serialisers already reduce
each instance to one pipeline at a time, so a second instance only helps
*simultaneous* users, at triple the memory. To scale horizontally the job
registry needs a shared store (GCS or Redis) instead of a dict — a real change
to `app.py`, which is exactly what is being kept verbatim.

Queue behaviour is the original's: `app.py` holds a `deque` and reports
`queue_position` from `/status/{id}`, which the page shows as its progress
message.

## Measured

Service: <https://prism-379257840013.asia-south1.run.app> (revision `prism-00007-4vm`)

| | |
|---|---|
| Image size | **684.2 MB** |
| Cold start (container ready) | **26.8 s** — 3.7 s back-conversion + 23.1 s warm-up |
| Cold `GET /` (first hit after scale-to-zero) | **27.9 s** |
| Warm `GET /` | **0.29 s** |
| Warm parse, per page | **14.0 s** median (13.92 / 14.09) |
| Parse right after a cold page load | **13.8 s** — already warm |
| `/health` + `/progress` while a parse runs | **389 ms** median |
| Peak RSS during a parse | **~2.0 GB** (1965-1998 MB) |
| Startup peak RSS | 1575-1720 MB |
| 2-page PDF (API-only revision) | 107.3 s (~53.7 s/page; dense maths) |

**The UI absorbs the cold start.** On the API-only revision the first request
after scale-to-zero cost 41.8 s (25.6 s start + a 14 s page). With the UI the
page load pays the 27.9 s, and by the time anyone has picked a file the
instance is warm, so the parse itself is 13.8 s. Cold start is now paid where a
spinner is expected rather than in the middle of a conversion.

Cold start is measured by idling 17 minutes to force scale to zero, then
timing the first request; the logs confirm a fresh container
(`READY in 25.88s`) rather than a reused instance.

Output parity against the same files run locally on Windows, on the API-only
revision:

| file | local | Cloud Run | identical |
|---|---|---|---|
| `ieee_p4_twocol_figure.png` | 3407 chars | 3407 chars | **byte-for-byte** |
| 2-page PDF | 13891 chars | 13891 chars | **byte-for-byte** |

`deploy/client_check.py` runs these checks:

```bash
python deploy/client_check.py "$URL" compare page.png local.md
python deploy/client_check.py "$URL" parse   doc.pdf --out out.md
```

## Notes

- **Python 3.12, not 3.11.** `pyproject.toml` sets `requires-python = ">=3.12"`
  and `.python-version` pins 3.12.
- **Two venvs.** `/opt/prism` (main) and `/app/venvs/rtable` (RapidTable child),
  separate because `rapid_table` pins a conflicting `rapidocr` major version.
  `onnx` is installed in **both** — omitting it from the child is what silently
  broke SLANet during the fp16 sweep, dropping tables to the coordinate
  heuristic with no error.
- **SLANet-plus is pre-fetched at build time**, so nothing downloads at start.
- **`perl` is a real dependency**, not incidental: `tlmgr` is a Perl program
  and `python:3.12-slim` ships only `perl-base`. `fontconfig` likewise —
  fontspec resolves `Noto Sans CJK SC` by name through `fc-list`.
- **The UI files are never edited.** `app.py` and `web/index.html` are copied
  into the image byte-for-byte from `d91c6b1` and `2e62b83`. Every container
  fix is a symlink, a `PATH` entry or a line in `serve.py` — so
  `git hash-object` on either file still matches its original commit, and
  that is the check to run if the layout ever looks wrong again.
- `libgl1`, `libglib2.0-0` (and `libgomp1`, `libsm6`, `libxext6`, `libxrender1`)
  are required or `import cv2` / onnxruntime fail with unhelpful loader errors.
- **`.gcloudignore` is load-bearing.** `gcloud builds submit` does not read
  `.dockerignore`; without a `.gcloudignore` it derives the upload context from
  `.gitignore`, under which the fp16 graphs are ignored — so the build would
  fail at `COPY`. It must also re-include the `Dockerfile` itself (a local
  `docker build` always sends it; the remote build does not), and must exclude
  `benchmarks/compare`, whose paths are deep enough to crash gcloud on Windows
  MAX_PATH. Keep it in sync with `.dockerignore`.
- **The in-build smoke test is not optional.** Two Linux-port failures got past
  every static existence check and were only caught by running a page:
  `math_worker_onnx.py` loads its tokenizer from `Texo/model/` (the parent of
  `onnx/`), and `run_omnidocbench.py` gates the layout detector on the presence
  of the *old* plus-L model this image replaces — its off-path opens a layout
  cache whose default path is `''`. Hence `PRISM_USE_PPDL_LAYOUT=1`, set both
  in the image and at `serve.py` import so the smoke test and the server share
  one configuration.
