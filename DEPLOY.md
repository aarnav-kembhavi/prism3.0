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
    --concurrency 1 \
    --timeout 900 \
    --min-instances 0 \
    --max-instances 3 \
    --cpu-boost \
    --allow-unauthenticated
```

| Flag | Why |
|---|---|
| `--memory 4Gi` | Startup peaks well above 2 GiB once the fp32 graphs are materialised; 2 GiB OOMs |
| `--cpu 4` | The pipeline is CPU-bound end to end |
| `--concurrency 1` | **Critical.** Default is 80. One request holds GBs of resident model and saturates the CPU, so anything above 1 OOMs the instance |
| `--timeout 900` | A dense multi-page PDF takes minutes |
| `--min-instances 0` | Scale to zero; cold start is the trade-off |
| `--max-instances 3` | Cost ceiling while testing |
| `--cpu-boost` | Model load is CPU-bound, and it all happens before the first request |
| `--allow-unauthenticated` | Public endpoint |

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

## Measured

Service: <https://prism-379257840013.asia-south1.run.app> (revision `prism-00003-flm`)

| | |
|---|---|
| Image size | **684.2 MB** |
| Cold start (container ready) | **25.6 s** — 2.6-3.7 s back-conversion + ~22 s warm-up |
| Cold first request, end to end | **41.8 s** (25.6 s start + 14.0 s page) |
| Warm, per page | **17.4 s** median (17.24 / 17.40 / 17.66); 14.0-17.0 s observed range |
| Peak RSS during a request | **1796 MB** |
| Startup peak RSS | 1546-1557 MB |
| Back-conversion peak RSS | 334-491 MB |
| 2-page PDF | 107.3 s (~53.7 s/page; dense math pages) |

Cold start was measured by leaving the service idle for 17 minutes so it scaled
to zero, then timing one request; the logs confirm a fresh container
(`READY in 25.58s`) rather than a reused instance. `--cpu-boost` matters here:
model load is entirely CPU-bound and happens before the port is served.

Output parity against the same files run locally on Windows:

| file | local | Cloud Run | identical |
|---|---|---|---|
| `ieee_p4_twocol_figure.png` | 3407 chars | 3407 chars | **byte-for-byte** |
| 2-page PDF | 13891 chars | 13891 chars | **byte-for-byte** |

Cloud Run is roughly 2x slower per page than the 16-core dev machine (8.5 s
local vs 17.4 s on 4 vCPU), which is the expected shape for a CPU-bound
pipeline.

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
