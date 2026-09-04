"""
PRISM HTTP service for Cloud Run.

    POST /parse    PDF or image upload -> markdown
    GET  /health   200 only once the models are loaded and a warm-up page ran

Design notes that are not obvious:

* **fp16 on disk, fp32 in the workers, converted once.** The image ships only
  the `_fp16.onnx` graphs (about half the bytes). The pipeline runs its models
  in *worker subprocesses*, so an in-process back-conversion in this server
  would not reach them -- each worker would redo it, per request. Instead
  startup materialises real fp32 graphs into PRISM_FP32_CACHE once and points
  the pipeline at those files via the per-graph PRISM_FP16_<KEY> override.
  The workers then open ordinary fp32 graphs and never convert anything.
  Note the materialised files must NOT be named `*_fp16.onnx`, or the
  load-time patch in pipeline/fp16_runtime.py would try to convert them again.

* **Workers are persistent.** benchmarks/run_omnidocbench.py:_run_prism_on_images
  is the pipeline's in-process entry point, but it starts workers, runs, and
  stops them. Re-running it per request would reload every ONNX graph per
  request. Rather than reimplement its ~100-line per-page body -- the risky
  part -- this module swaps the two worker classes for factories that hand back
  a started singleton and ignore stop(). The harness code then runs verbatim.

* **Cloud Run specifics.** Bind 0.0.0.0 on $PORT (default 8080) or the
  container is killed. Request bodies are capped at 32 MB upstream, so uploads
  over 30 MB are refused explicitly rather than truncated silently.
"""
import io
import os
import sys
import tempfile
import threading
import time
import shutil
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("prism")

MAX_UPLOAD_BYTES = 30 * 1024 * 1024        # Cloud Run caps the body at 32 MB
FP32_CACHE = Path(os.environ.get("PRISM_FP32_CACHE", "/tmp/prism_fp32"))

# The measured-best configuration from the fp16 sweep (quant/README.md):
# every graph fp16 EXCEPT slanet-plus, which is the only one that costs real
# quality (mean similarity 0.937 vs 0.996 for the rest).
FP16_GRAPHS = ["texo_encoder", "texo_decoder", "ppocr_rec",
               "ppocr_det", "ppocr_rec_en", "ppdoclayout_v3"]

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

_ready = False
_ready_error: str | None = None
_run_lock = threading.Lock()
_startup_stats: dict = {}


# ── memory sampling ──────────────────────────────────────────────────────────

def _tree_rss_mb() -> float:
    """RSS of this process plus every worker subprocess, in MB."""
    import psutil
    proc = psutil.Process()
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except Exception:
            pass
    return total / 1e6


class PeakRSS:
    """Sample process-tree RSS in the background and keep the maximum."""

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        def loop():
            while not self._stop.is_set():
                try:
                    self.peak = max(self.peak, _tree_rss_mb())
                except Exception:
                    pass
                self._stop.wait(self.interval)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        try:
            self.peak = max(self.peak, _tree_rss_mb())
        except Exception:
            pass
        return False


# ── startup: fp16 -> fp32, once ──────────────────────────────────────────────

def materialize_fp32_graphs() -> dict:
    """
    Convert every baked fp16 graph back to fp32 on disk, exactly once.

    Returns a summary for the startup log. Writes files whose names match the
    ORIGINAL fp32 graph, so nothing downstream mistakes them for fp16.
    """
    from pipeline.quant_select import GRAPHS, fp16_path
    from pipeline.fp16_runtime import fp32_bytes_from_fp16

    FP32_CACHE.mkdir(parents=True, exist_ok=True)
    done, skipped, total_mb = [], [], 0.0

    for key in FP16_GRAPHS:
        src = fp16_path(key)
        if not src.exists():
            skipped.append(f"{key}: no fp16 graph at {src.name}")
            continue
        dst = FP32_CACHE / Path(GRAPHS[key]).name        # original fp32 filename
        blob, stats = fp32_bytes_from_fp16(src)
        dst.write_bytes(blob)
        # Per-graph override: graph_path() returns this file, and because the
        # name does not end in _fp16.onnx the load-time patch leaves it alone.
        os.environ[f"PRISM_FP16_{key.upper()}"] = str(dst)
        mb = dst.stat().st_size / 1e6
        total_mb += mb
        done.append(f"{key} {src.stat().st_size/1e6:.1f}->{mb:.1f}MB casts={stats['casts']}")

    return {"converted": done, "skipped": skipped,
            "materialized_mb": round(total_mb, 1)}


def _make_workers_persistent():
    """
    Replace the two worker classes with singleton factories.

    _run_prism_on_images() constructs a worker, start()s it, and stop()s it on
    the way out. For a server that means reloading every ONNX graph on every
    request. These factories return one already-started instance and turn
    stop() into a no-op, so the harness body is untouched but the models stay
    resident between requests.
    """
    import pipeline.text_worker as tw
    import pipeline.math_worker_onnx as mw

    singletons: dict = {}
    shutdown = []

    def wrap(module, name):
        real_cls = getattr(module, name)

        def factory(*args, **kwargs):
            inst = singletons.get(name)
            if inst is None:
                inst = real_cls(*args, **kwargs)
                real_start, real_stop = inst.start, inst.stop

                def start_once(_i=inst, _s=real_start):
                    if not getattr(_i, "_prism_started", False):
                        _s()
                        _i._prism_started = True

                inst.start = start_once
                inst.stop = lambda: None
                shutdown.append(real_stop)
                singletons[name] = inst
            return inst

        setattr(module, name, factory)

    # PRISM_SINGLE_WORKER=1 selects the product config (1 OCR + 1 math), which
    # is what orchestrate.py uses and what suits 4 CPUs at concurrency 1.
    os.environ.setdefault("PRISM_SINGLE_WORKER", "1")
    wrap(tw, "TextOCRWorker")
    wrap(mw, "MathOCRWorkerOnnx")
    return shutdown


_worker_shutdown: list = []


def warm_up() -> dict:
    """Load every model by running one real page through the pipeline."""
    img = None
    baked = ROOT / "deploy" / "warmup.png"          # what ships in the image
    if baked.exists():
        img = baked
    else:                                            # local checkout fallback
        for cand in ["test_images/real/clean", "test_images/real/misc"]:
            d = ROOT / cand
            if d.is_dir():
                for f in sorted(d.iterdir()):
                    if f.suffix.lower() in IMAGE_SUFFIXES:
                        img = f
                        break
            if img:
                break
    if img is None:
        raise RuntimeError(
            "no warm-up image: expected deploy/warmup.png in the image, or a "
            "page under test_images/ in a local checkout")

    t0 = time.perf_counter()
    md = run_pipeline([str(img)])
    return {"warmup_image": img.name,
            "warmup_s": round(time.perf_counter() - t0, 2),
            "warmup_chars": len(md[0] if md else "")}


def run_pipeline(image_paths: list[str]) -> list[str]:
    """Run PRISM over page images, in order. Returns one markdown per page."""
    from benchmarks.run_omnidocbench import _run_prism_on_images

    work = Path(tempfile.mkdtemp(prefix="prism_req_"))
    try:
        with _run_lock:                       # workers are not re-entrant
            out = _run_prism_on_images([str(p) for p in image_paths], str(work))
        return [out.get(Path(p).stem, "") for p in image_paths]
    finally:
        shutil.rmtree(work, ignore_errors=True)


def rasterize_pdf(data: bytes, dpi: int = 200) -> list[str]:
    """PDF bytes -> one PNG per page on disk. Returns the paths, in order."""
    import pypdfium2 as pdfium

    out_dir = Path(tempfile.mkdtemp(prefix="prism_pdf_"))
    doc = pdfium.PdfDocument(io.BytesIO(data))
    paths = []
    try:
        for i in range(len(doc)):
            page = doc[i]
            pil = page.render(scale=dpi / 72).to_pil().convert("RGB")
            p = out_dir / f"page_{i + 1:04d}.png"
            pil.save(p)
            paths.append(str(p))
    finally:
        doc.close()
    return paths


# ── FastAPI app ──────────────────────────────────────────────────────────────

from fastapi import FastAPI, File, HTTPException, UploadFile   # noqa: E402
from fastapi.responses import JSONResponse, PlainTextResponse  # noqa: E402

app = FastAPI(title="PRISM", docs_url=None, redoc_url=None)


@app.on_event("startup")
def _startup() -> None:
    global _ready, _ready_error, _worker_shutdown
    try:
        t0 = time.perf_counter()
        with PeakRSS() as conv_rss:
            conv = materialize_fp32_graphs()
        conv_s = time.perf_counter() - t0
        log.info("fp16->fp32 back-conversion done in %.2fs | peak RSS %.0f MB | %s",
                 conv_s, conv_rss.peak, "; ".join(conv["converted"]) or "nothing")
        for s in conv["skipped"]:
            log.warning("back-conversion skipped: %s", s)

        _worker_shutdown = _make_workers_persistent()

        with PeakRSS() as warm_rss:
            warm = warm_up()
        log.info("warm-up ran %s in %.2fs (%d chars) | peak RSS %.0f MB",
                 warm["warmup_image"], warm["warmup_s"], warm["warmup_chars"],
                 warm_rss.peak)

        _startup_stats.update(
            back_conversion_s=round(conv_s, 2),
            back_conversion_peak_rss_mb=round(conv_rss.peak),
            materialized_mb=conv["materialized_mb"],
            warmup_s=warm["warmup_s"],
            startup_peak_rss_mb=round(max(conv_rss.peak, warm_rss.peak)),
            total_startup_s=round(time.perf_counter() - t0, 2),
        )
        log.info("READY in %.2fs | startup peak RSS %d MB",
                 _startup_stats["total_startup_s"],
                 _startup_stats["startup_peak_rss_mb"])
        _ready = True
    except Exception as exc:                     # keep /health failing, loudly
        import traceback
        _ready_error = f"{type(exc).__name__}: {exc}"
        log.error("STARTUP FAILED: %s\n%s", _ready_error, traceback.format_exc())


@app.on_event("shutdown")
def _shutdown() -> None:
    for stop in _worker_shutdown:
        try:
            stop()
        except Exception:
            pass


@app.get("/health")
def health():
    if not _ready:
        raise HTTPException(503, _ready_error or "models still loading")
    return JSONResponse({"status": "ok", **_startup_stats})


@app.post("/parse")
async def parse(file: UploadFile = File(...)):
    if not _ready:
        raise HTTPException(503, _ready_error or "models still loading")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in IMAGE_SUFFIXES and suffix != ".pdf":
        raise HTTPException(
            400, f"Unsupported file type {suffix!r}. "
                 f"Send a PDF or one of {sorted(IMAGE_SUFFIXES)}.")

    # Read with a hard cap. Cloud Run refuses bodies over 32 MB, and a silent
    # truncation would look like a corrupt document rather than a rejection.
    data = bytearray()
    while chunk := await file.read(1 << 20):
        data.extend(chunk)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                413, f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024*1024)} MB "
                     f"limit (Cloud Run caps request bodies at 32 MB).")
    if not data:
        raise HTTPException(400, "Empty upload.")

    tmp_paths: list[str] = []
    tmp_dir = None
    t0 = time.perf_counter()
    try:
        with PeakRSS() as rss:
            if suffix == ".pdf":
                tmp_paths = rasterize_pdf(bytes(data))
                if not tmp_paths:
                    raise HTTPException(400, "PDF contains no pages.")
                tmp_dir = Path(tmp_paths[0]).parent
            else:
                tmp_dir = Path(tempfile.mkdtemp(prefix="prism_img_"))
                p = tmp_dir / f"upload{suffix}"
                p.write_bytes(bytes(data))
                tmp_paths = [str(p)]

            pages = run_pipeline(tmp_paths)

        wall = time.perf_counter() - t0
        log.info("parse ok | file=%s pages=%d bytes=%d wall=%.2fs "
                 "per_page=%.2fs peak_rss=%.0fMB",
                 file.filename, len(tmp_paths), len(data), wall,
                 wall / max(len(tmp_paths), 1), rss.peak)

        body = "\n\n".join(
            (f"<!-- page {i + 1} -->\n{md}" if len(pages) > 1 else md)
            for i, md in enumerate(pages))
        return PlainTextResponse(body, media_type="text/markdown; charset=utf-8")
    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        log.error("parse failed | file=%s wall=%.2fs\n%s",
                  file.filename, time.perf_counter() - t0, traceback.format_exc())
        raise HTTPException(500, f"{type(exc).__name__}: {exc}")
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    # Cloud Run injects PORT and kills the container if it binds anything else.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")),
                workers=1, timeout_keep_alive=65)
