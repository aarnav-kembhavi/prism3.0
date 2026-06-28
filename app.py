"""
PRISM web UI — FastAPI backend.

Endpoints:
  GET  /              → serve index.html
  POST /upload        → accept image, queue job, return {job_id}
  GET  /status/{id}   → {status, message, queue_position}
  GET  /pdf/{id}      → PDF bytes (when done)
"""

import sys
import threading
import time
import uuid
import shutil
import subprocess
from collections import deque
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

ROOT       = Path(__file__).parent
UPLOAD_DIR = ROOT / "_web_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI()

# ── Job registry ──────────────────────────────────────────────────────────────

_jobs: dict[str, dict] = {}
_queue: deque[str]     = deque()
_lock                  = threading.Lock()
_worker_busy           = False

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


# ── Background workers ────────────────────────────────────────────────────────

def _cleanup(job_id: str, delay: int = 600):
    time.sleep(delay)
    job = _jobs.pop(job_id, None)
    if not job:
        return
    shutil.rmtree(job.get("output_dir") or "", ignore_errors=True)
    up = job.get("upload_path")
    if up:
        Path(up).unlink(missing_ok=True)


def _process_job(job_id: str) -> None:
    global _worker_busy
    job = _jobs[job_id]
    try:
        image_path = Path(job["upload_path"])
        stem       = image_path.stem

        # ── Stage 1: OCR pipeline ──────────────────────────────────────────
        job["message"] = "Running OCR pipeline…"
        result = subprocess.run(
            [sys.executable, str(ROOT / "pipeline" / "orchestrate.py"), str(image_path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )

        output_dir = ROOT / "outputs" / f"{stem}_output"
        tex_path   = output_dir / "main.tex"

        if not tex_path.exists():
            tail = (result.stderr or result.stdout or "no output")[-600:]
            job.update(status="error", message=f"Pipeline failed:\n{tail}")
            return

        job["output_dir"] = str(output_dir)

        # ── Stage 2: LaTeX → PDF ───────────────────────────────────────────
        job["message"] = "Compiling PDF…"
        compiler = "xelatex" if "\\usepackage{xeCJK}" in tex_path.read_text(encoding="utf-8") else "pdflatex"
        subprocess.run(
            [compiler, "-interaction=nonstopmode", "main.tex"],
            cwd=str(output_dir),
            capture_output=True,
            timeout=120,
        )

        pdf_path = output_dir / "main.pdf"
        if not pdf_path.exists():
            job.update(status="error", message="PDF compilation failed.")
            return

        job.update(status="done", message="Done.", pdf_path=str(pdf_path))
        threading.Thread(target=_cleanup, args=(job_id,), daemon=True).start()

    except subprocess.TimeoutExpired:
        job.update(status="error", message="Timed out after 5 minutes.")
    except Exception as exc:
        job.update(status="error", message=str(exc))
    finally:
        with _lock:
            _worker_busy = False
        _pump_queue()


def _pump_queue() -> None:
    global _worker_busy
    with _lock:
        if _worker_busy or not _queue:
            return
        job_id       = _queue.popleft()
        _worker_busy = True
    _jobs[job_id]["status"] = "processing"
    threading.Thread(target=_process_job, args=(job_id,), daemon=True).start()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower() or ".png"
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    job_id   = uuid.uuid4().hex
    img_path = UPLOAD_DIR / f"{job_id}{suffix}"
    img_path.write_bytes(await file.read())

    _jobs[job_id] = {
        "status":      "queued",
        "message":     "Queued…",
        "upload_path": str(img_path),
        "output_dir":  None,
        "pdf_path":    None,
    }
    with _lock:
        _queue.append(job_id)
    _pump_queue()

    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    queue_pos = None
    if job["status"] == "queued":
        try:
            queue_pos = list(_queue).index(job_id) + 1
        except ValueError:
            pass

    return JSONResponse({
        "status":         job["status"],
        "message":        job["message"],
        "queue_position": queue_pos,
    })


@app.get("/pdf/{job_id}")
def get_pdf(job_id: str):
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "PDF not ready")
    pdf = Path(job["pdf_path"])
    if not pdf.exists():
        raise HTTPException(404, "PDF file missing")
    from fastapi.responses import Response
    return Response(
        pdf.read_bytes(),
        media_type="application/pdf",
        headers={"Content-Disposition": 'inline; filename="output.pdf"'},
    )


@app.get("/latex/{job_id}")
def get_latex(job_id: str):
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "LaTeX not ready")
    tex = Path(job["output_dir"]) / "main.tex"
    if not tex.exists():
        raise HTTPException(404, "LaTeX file missing")
    from fastapi.responses import Response
    return Response(tex.read_text(encoding="utf-8"), media_type="text/plain; charset=utf-8")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
