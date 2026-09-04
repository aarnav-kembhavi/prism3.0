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
