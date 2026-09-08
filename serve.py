"""
PRISM HTTP service for Cloud Run.

    GET  /          the PRISM web UI            } app.py, verbatim
    POST /upload    image -> job id                }   (web/index.html @ 2e62b83,
    GET  /status/{id}  job state                   }    app.py @ d91c6b1); this
    GET  /pdf/{id}     compiled PDF                }    module registers none of
    GET  /latex/{id}   main.tex                    }    them itself
    POST /parse     PDF or image upload -> markdown   } this module
    GET  /health    200 only once models are loaded   }
    GET  /progress  what the server is doing now      }

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
import asyncio
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


def _configure_env() -> None:
    """
    Pipeline flags this deployment cannot leave at their defaults.

    run_omnidocbench.py decides whether to use the layout detector at all by
    testing for the OLD models/ppdoclayout/ppdoclayout_plus_l.onnx, which this
    image does not ship. The default therefore resolves to "off", and the
    off-path opens a pre-computed layout cache whose default path is the empty
    string -- FileNotFoundError: ''. Caught by the in-build smoke test.

    Applied at import so deploy/smoke_test.py runs the same configuration the
    server does, rather than only the server being correct.
    """
    os.environ.setdefault("PRISM_USE_PPDL_LAYOUT", "1")
    os.environ.setdefault("PRISM_PPDL_V3", "1")
    os.environ.setdefault("PRISM_SINGLE_WORKER", "1")

    # In the image /app/_web_uploads and /app/outputs are symlinks into the
    # tmpfs, because app.py:25 and orchestrate.py:213 hardcode those two paths
    # and neither file is deployment code. A symlink whose target does not
    # exist is not a directory, so app.py's UPLOAD_DIR.mkdir(exist_ok=True)
    # would raise at import -- create the targets first. Only in the image:
    # a local checkout has real directories and no /tmp.
    if os.name == "posix" and str(ROOT) == "/app":
        for link in (ROOT / "_web_uploads", ROOT / "outputs"):
            if link.is_symlink():
                os.makedirs(os.readlink(str(link)), exist_ok=True)


_configure_env()

_ready = False
_ready_error: str | None = None
_run_lock = threading.Lock()
_startup_stats: dict = {}

# One parse at a time, enforced HERE rather than by Cloud Run --concurrency.
#
# A parse holds ~1.8 GB resident, and the materialised fp32 graphs occupy
# another 251 MB of tmpfs against the same limit, so two concurrent parses
# would exceed 4Gi. Cloud Run's --concurrency would enforce that, but it is
# far too blunt once a UI is attached: at --concurrency 1 the page itself, its
# assets, /health and every progress poll all queue behind a 17-second parse,
# and the app looks hung. Cloud Run runs at --concurrency 4 so those pass
# straight through; this semaphore is what actually protects memory.
_parse_sem = asyncio.Semaphore(1)

# Live state for the UI. There is no live stage information to be had from the
# pipeline through this entry point, so the UI is given only things that are
# actually true: whether a parse is running or queued behind one, how many are
# waiting, and which page of a multi-page PDF is in flight.
_progress: dict = {"state": "idle", "page": 0, "pages": 0,
                   "started": None, "waiting": 0}
_progress_lock = threading.Lock()


def _set_progress(**kw) -> None:
    with _progress_lock:
        _progress.update(kw)


def _progress_snapshot() -> dict:
    with _progress_lock:
        snap = dict(_progress)
    started = snap.pop("started", None)
    snap["elapsed_s"] = round(time.perf_counter() - started, 1) if started else 0.0
    return snap


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


def run_pipeline(image_paths: list[str], on_page=None) -> list[str]:
    """
    Run PRISM over page images, in order. Returns one markdown per page.

    Called once PER PAGE rather than once for the whole list, so multi-page
    PDFs can report real progress ("page 3 of 7") instead of a spinner that
    sits still for two minutes. The workers are persistent singletons, so the
    per-call cost is a scratch directory and a sampler thread, not a model
    reload.
    """
    from benchmarks.run_omnidocbench import _run_prism_on_images

    out: list[str] = []
    for i, p in enumerate(image_paths):
        if on_page:
            on_page(i, len(image_paths))
        work = Path(tempfile.mkdtemp(prefix="prism_req_"))
        try:
            with _run_lock:                   # workers are not re-entrant
                res = _run_prism_on_images([str(p)], str(work))
            out.append(res.get(Path(p).stem, ""))
        finally:
            shutil.rmtree(work, ignore_errors=True)
    if on_page:
        on_page(len(image_paths), len(image_paths))
    return out


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

# The UI and its routes come from app.py as committed at d91c6b1 (2026-07-14),
# serving web/index.html as committed at 2e62b83 (2026-06-29). Neither file is
# modified: app.py is imported and its router spliced in whole, so GET /,
# /upload, /status/{id}, /pdf/{id} and /latex/{id} are the originals. It is
# included FIRST so its GET / wins -- this module registers no "/" of its own.
#
# app.py runs the pipeline as an `orchestrate.py` subprocess per job, which
# inherits os.environ and therefore the PRISM_FP16_<KEY> overrides that
# materialize_fp32_graphs() sets. That is the whole reason it works here
# unchanged.
import app as legacy                                              # noqa: E402

# Only app.py's OWN routes. legacy.app.router also carries the /docs,
# /redoc and /openapi.json routes FastAPI attaches to every app, and this
# service disables those deliberately (docs_url=None above).
for _route in legacy.app.router.routes:
    if getattr(getattr(_route, "endpoint", None), "__module__", None) == "app":
        app.router.routes.append(_route)


def _claim_legacy_worker():
    """
    Hold app.py's single-worker flag for the duration of a /parse.

    app.py serialises its own jobs with `_worker_busy`, and this module
    serialises /parse with a semaphore, but the two knew nothing about each
    other: a /upload job and a /parse could run two pipelines at once and blow
    through 4Gi. This claims app.py's flag using app.py's own protocol -- lock,
    test, set, and _pump_queue() on release -- rather than editing app.py.
    """
    import contextlib

    @contextlib.contextmanager
    def _guard():
        while True:
            with legacy._lock:
                if not legacy._worker_busy:
                    legacy._worker_busy = True
                    break
            time.sleep(0.2)
        try:
            yield
        finally:
            with legacy._lock:
                legacy._worker_busy = False
            legacy._pump_queue()

    return _guard()


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


@app.get("/progress")
def progress():
    """
    What the server is actually doing right now.

    Cheap and outside the semaphore, so it answers instantly while a parse
    holds the lock -- that is the whole point of running Cloud Run at
    --concurrency 4 rather than 1.
    """
    return JSONResponse({"ready": _ready, **_progress_snapshot()})


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
    queued = False
    try:
        # Only the parse itself is serialised. Everything above -- upload,
        # validation, size check -- and every other route stays free.
        if _parse_sem.locked():
            queued = True
            _set_progress(waiting=_progress_snapshot().get("waiting", 0) + 1)
            log.info("parse queued | file=%s", file.filename)
        async with _parse_sem:
            if queued:
                _set_progress(waiting=max(_progress_snapshot().get("waiting", 1) - 1, 0))
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

                _set_progress(state="running", page=0, pages=len(tmp_paths),
                              started=time.perf_counter())

                def _on_page(done, total):
                    _set_progress(state="running", page=done + 1 if done < total else total,
                                  pages=total)

                # The pipeline is synchronous and CPU-bound; run it off the
                # event loop so /progress and /health keep responding.
                def _guarded():
                    with _claim_legacy_worker():
                        return run_pipeline(tmp_paths, _on_page)

                pages = await asyncio.to_thread(_guarded)

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
        _set_progress(state="idle", page=0, pages=0, started=None)
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    # Cloud Run injects PORT and kills the container if it binds anything else.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")),
                workers=1, timeout_keep_alive=65)
