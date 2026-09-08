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
IMG="asia-south1-docker.pkg.dev/$PROJECT/prism/prism:v7"   # currently deployed

gcloud builds submit --tag "$IMG" --timeout=3600s --region=asia-south1
```

The build runs `deploy/texcheck.py` and then `deploy/smoke_test.py`, so a TeX
install that cannot compile, or a UI that is not the original page, fails the
build rather than the deploy. Expect about 7 minutes.

## Deploy

Every flag below is non-default and load-bearing:

```bash
gcloud run deploy prism \
    --image "$IMG" \
    --region asia-south1 \
    --memory 8Gi \
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
| `--memory 8Gi` | A `/upload` job measured 2687 MB peak -- `orchestrate.py` loads a second full model set alongside the resident workers -- plus 251 MB of tmpfs graphs. 4Gi fit that with ~1.2 GB spare, which is too thin a margin to hold across page types; 8Gi is headroom bought deliberately rather than tuned |
| `--cpu 4` | The pipeline is CPU-bound end to end |
| `--concurrency 4` | Page loads, `/health` and every `/status/{id}` poll must not queue behind a running job. Memory is protected by an `asyncio.Semaphore(1)` around the parse itself, not by this number — see Concurrency below |
| `--timeout 900` | A dense multi-page PDF takes minutes |
| `--min-instances 0` | Scale to zero; cold start is the trade-off |
| `--max-instances 1` | **Not a cost knob.** `app.py`'s job registry is a per-instance dict; with more than one instance a `/status/{id}` poll can land on an instance that never saw the upload and the job is lost — see Concurrency below |
| `--cpu-boost` | Model load is CPU-bound, and it all happens before the first request |
| `--no-cpu-throttling` | **Load-bearing, 18x.** `app.py` hands each job to a detached background thread and returns immediately, so with Cloud Run's default (CPU only during a request) the instance is throttled for the entire pipeline run: the same page took 435 s throttled and 23.9 s unthrottled. Costs CPU for the instance's whole lifetime rather than per request — see Measured |
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
memory limit. That, plus the second model set a `/upload` job spawns, is why
`--memory` is 8Gi.

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
`/upload` job and a `/parse` could have run two pipelines at once, doubling the
2.7 GB high-water mark. `serve.py` now claims `app.py`'s flag using its own
protocol -- lock, test, set, `_pump_queue()` on release -- rather than editing it.

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
251 MB of tmpfs against the same limit, and a `/upload` job peaks at 2.7 GB, so
concurrent pipelines are still serialised on purpose rather than left to the
memory limit. `--concurrency 1` enforces that but is the wrong tool once a UI
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

Service: <https://prism-379257840013.asia-south1.run.app> (revision `prism-00009-qsn`)

| | v6 (no TeX) | **v7** |
|---|---|---|
| Image size, compressed | 684.2 MB | **820.4 MB** |
| TeX cost | — | **+136.2 MB** |
| Cold start (container ready) | 26.8 s | **23.6 s** — 2.5 s back-conversion + 21.1 s warm-up |
| Cold `GET /` (first hit after scale-to-zero) | 27.9 s | **25.7 s** |
| Warm `GET /` | 0.29 s | **0.33 s** |
| Startup peak RSS | 1575-1720 MB | **1530-1590 MB** |

**The TeX did not cost cold start.** 820 MB pulls no slower than 684 MB here,
and startup does the same work either way — nothing TeX-related runs before the
port is served. Cold start is measured by idling 18 minutes to force scale to
zero, then timing the first request; the logs confirm a fresh container
(`READY in 23.60s`) rather than a reused instance.

On disk inside the image: TinyTeX **233 MB** (129 packages), Noto Sans CJK
**38 MB** after deleting the Serif family. For comparison, the Debian route
(`texlive-xetex` + `texlive-latex-extra` + `texlive-lang-chinese` +
`fonts-noto-cjk`) was estimated at 1.8-2.3 GB — roughly 15x this.

### Reproducing the image size

`gcloud artifacts docker images list --format='...imageSizeBytes'` returns **0**
for these images, so the sizes above come from summing the manifest instead —
config plus every layer, which is the compressed pull size:

```bash
PROJECT=$(gcloud config get-value project)
REPO="asia-south1-docker.pkg.dev/$PROJECT/prism/prism"
TAG=v7

DIG=$(gcloud artifacts docker images list "$REPO" --include-tags --format=json \
      | python -c "import sys,json,os;t=os.environ['TAG'];print(next(i['version'] for i in json.load(sys.stdin) if t in (i.get('tags') or [])))")

curl -s -H "Authorization: Bearer $(gcloud auth print-access-token)" \
     -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
     "https://$(echo $REPO | cut -d/ -f1)/v2/$PROJECT/prism/prism/manifests/$DIG" \
  | python -c "import sys,json;m=json.load(sys.stdin);print('%.1f MB in %d layers' % ((m['config']['size']+sum(l['size'] for l in m['layers']))/1e6, len(m['layers'])))"
```

v6 reports `684.2 MB in 35 layers`, v7 `820.4 MB in 44 layers`. The
uncompressed figures quoted above (TinyTeX 233 MB, Noto 38 MB) are `du -sh`
inside the build, printed by the Dockerfile's TeX layers.

### One page, end to end through the UI

`POST /upload` → `GET /status/{id}` → `GET /pdf/{id}`, on
`ieee_p4_twocol_figure.png` (two-column, so the preamble emits `paracol`):

| | |
|---|---|
| Pipeline (`orchestrate.py` subprocess) | **20.9 s** |
| LaTeX → PDF (`pdflatex`) | **2.3 s** |
| Total, upload to `done` | **23.9 s** |
| PDF | 312 207 bytes, 1 page, 3448 text chars |
| `main.tex` | 4043 chars |

Preamble actually served: `amsmath, booktabs, geometry, graphicx, inputenc,
paracol, ragged2e` — `paracol` present, so the visual-fidelity two-column path
(`PRISM_VISUAL_FIDELITY=1`, set by `app.py:64` for every web job) compiles.

Peak RSS during a `/upload` job, measured in the build: **2687 MB**. That is
the deployment's real high-water mark — `orchestrate.py` loads its own full
model set while `serve.py` still holds the persistent workers resident. Plus
251 MB of materialised graphs in tmpfs, about 2.9 GB. That fits inside 4Gi, but
only with ~1.2 GB to spare, so the service is deployed at **8Gi** to give the
second model set room without anyone tuning it per page type.

### `--no-cpu-throttling` is not optional here

The same page took **435 s** on the first deployment and **23.9 s** after
adding `--no-cpu-throttling`: **18x**. The PDF was byte-identical both times
(312 207 bytes), so this is purely CPU starvation, not different work.

Cloud Run's default allocates CPU only while a request is in flight. `app.py`
returns the `job_id` immediately and does the work in a **detached background
thread**, so from the platform's side nothing is in flight for the whole
pipeline run and the instance drops to a sliver of CPU. The 2 s `/status` polls
were the only thing giving it any CPU at all.

This is not tunable from the app layer without moving `app.py`'s worker into
the request — which is the file being kept verbatim. The v6 API did not have
the problem because `/parse` does its work *inside* the request
(`await asyncio.to_thread(...)`), so CPU stayed allocated throughout.

The cost: CPU is billed for an instance's whole lifetime rather than per
request. `--min-instances 0` still applies, so it scales to zero when idle.

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
