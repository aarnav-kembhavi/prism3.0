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
    --max-instances 3 \
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
| `--max-instances 3` | Cost ceiling while testing |
| `--cpu-boost` | Model load is CPU-bound, and it all happens before the first request |
| `--allow-unauthenticated` | Public endpoint |
| `--session-affinity` | `/progress` is per-instance state; without affinity a poll can land on a different instance than the parse and report `idle` |

## Use

```bash
URL=$(gcloud run services describe prism --region asia-south1 \
      --format='value(status.url)')

curl -s "$URL/health"                                    # 200 only once loaded
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

`GET /` serves the same page `app.py` does, rebuilt for this service by
`deploy/build_ui.py` (markup and all ~10 KB of CSS kept verbatim; only the
client logic is replaced). `app.py`'s JS drives `/upload` -> `/status` ->
`/pdf`, which needs a LaTeX toolchain this image deliberately does not carry,
so the client targets `/parse` and renders the markdown with marked + KaTeX.

```bash
python deploy/build_ui.py        # web/index.html -> deploy/ui.html
```

## Concurrency

`--concurrency 4`, with an `asyncio.Semaphore(1)` around the parse only.

A parse holds ~1.8 GB resident and the materialised fp32 graphs take another
251 MB of tmpfs against the same limit, so two concurrent parses would exceed
4Gi. `--concurrency 1` enforces that but is the wrong tool once a UI exists:
the page, `/health` and every progress poll would queue behind a 17-second
parse and the app would look hung. Measured on the live service, with a parse
running: **389 ms median** across `/health` and `/progress`.

`--session-affinity` matters. `/progress` is per-instance state, so a poll that
lands on a different instance than the parse reports `idle`. Affinity pins a
client to one instance. Verified locally, where there is exactly one instance:
a second parse arriving 1.5 s into the first reports `waiting=1` for 10 s, then
starts with its own elapsed clock when the first finishes (11.9 s and 22.4 s
end to end).

## Measured

Service: <https://prism-379257840013.asia-south1.run.app> (revision `prism-00005-fxw`)

| | |
|---|---|
| Image size | **684.2 MB** |
| Cold start (container ready) | **25.9 s** — 3.8 s back-conversion + ~22 s warm-up |
| Cold `GET /` (first hit after scale-to-zero) | **27.9 s** |
| Warm `GET /` | **0.49 s** |
| Warm parse, per page | **13.6 s** median (13.56 / 13.57 / 14.19) |
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
