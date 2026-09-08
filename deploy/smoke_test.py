"""
End-to-end smoke test, run during the image build.

Exercises the real startup path -- fp16 -> fp32 materialisation, worker spawn,
one full page through the pipeline -- and then the ORIGINAL UI's own route
chain, POST /upload -> GET /status/{id} -> GET /pdf/{id}, ending in a PDF with
a real page count. Exits non-zero if anything is wrong, so a broken image fails
the build instead of failing at deploy time.

This exists because static existence checks are not enough. The first deployed
revision passed every "does the file exist" assertion and still died at
startup: math_worker_onnx.py reads its tokenizer from Texo/model/, the PARENT
of the onnx/ directory the graphs live in, and only actually running a page
surfaces that.

The GET / assertions below name the ids and routes that are actually in
web/index.html as committed at 2e62b83 -- #latex-pre, #pdf-viewer, /upload,
/status/ -- so that serving anything other than the original page fails here.

    PRISM_FP32_CACHE=/tmp/buildcheck python deploy/smoke_test.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import serve  # noqa: E402

MIN_CHARS = 200
JOB_TIMEOUT_S = 600


def check_ui(client) -> int:
    r = client.get("/")
    ctype = r.headers.get("content-type", "")
    print("GET /      -> %d %s, %d bytes" % (r.status_code, ctype, len(r.text)))
    if r.status_code != 200:
        print("FAIL: GET / returned %d" % r.status_code)
        return 1
    if "text/html" not in ctype:
        print("FAIL: GET / content-type is %r, expected text/html" % ctype)
        return 1

    # The original page, element for element. Anything served here that is not
    # web/index.html @ 2e62b83 will be missing at least one of these.
    missing = [m for m in (
        "<html",
        'id="drop-zone"', 'id="file-input"',
        'id="upload-panel"', 'id="processing-panel"', 'id="result-panel"',
        'id="latex-pre"', 'id="latex-code"',        # left pane: LaTeX Source
        'id="pdf-viewer"',                          # right pane: PDF Preview
        ">LaTeX Source<", ">PDF Preview<",
        "fetch('/upload'", "/status/${jobId}", "/pdf/${jobId}", "/latex/${jobId}",
    ) if m not in r.text]
    if missing:
        print("FAIL: the page served is not the original UI; missing %s" % missing)
        return 1

    # Two panes side by side is the original layout. A regression to one
    # full-width pane fails the build rather than shipping.
    panes = r.text.count('<div class="pane">')
    if r.text.count('<div class="split">') != 1 or panes != 2:
        print("FAIL: expected one .split with two .pane children, got "
              "%d split / %d panes" % (r.text.count('<div class="split">'), panes))
        return 1
    print("layout    -> .split with %d panes, original ids present" % panes)
    return 0


def check_job(client) -> int:
    """POST /upload through to a PDF, the way the UI does it."""
    img = Path(serve.ROOT) / "deploy" / "warmup.png"
    if not img.exists():
        print("FAIL: no upload fixture at %s" % img)
        return 1

    t0 = time.perf_counter()
    with open(img, "rb") as fh:
        r = client.post("/upload", files={"file": (img.name, fh, "image/png")})
    print("POST /upload -> %d %s" % (r.status_code, r.text[:200]))
    if r.status_code != 200:
        print("FAIL: /upload returned %d" % r.status_code)
        return 1
    job_id = r.json()["job_id"]

    # Peak RSS across the whole tree while the job runs. This is the number
    # that decides --memory: a /upload job spawns orchestrate.py, which loads
    # its OWN copy of every model, while serve.py is already holding the
    # persistent workers resident from warm-up. Two model sets at once is the
    # real high-water mark of this deployment, and nothing else measures it --
    # /health only reports startup.
    state = None
    peak = 0.0
    while time.perf_counter() - t0 < JOB_TIMEOUT_S:
        peak = max(peak, serve._tree_rss_mb())
        s = client.get("/status/%s" % job_id)
        if s.status_code != 200:
            print("FAIL: /status returned %d" % s.status_code)
            return 1
        state = s.json()
        if state["status"] in ("done", "error"):
            break
        time.sleep(2)
    print("peak RSS during /upload job: %.0f MB "
          "(persistent workers + orchestrate.py subprocess)" % peak)

    print("job %s -> %s (%.1fs) %s"
          % (job_id, state and state["status"], time.perf_counter() - t0,
             (state or {}).get("message", "")[:300]))
    if not state or state["status"] != "done":
        print("FAIL: job did not reach 'done'")
        return 1

    # The right pane is this response. A compile that produced no pages is a
    # broken image, not a runtime hiccup.
    p = client.get("/pdf/%s" % job_id)
    print("GET /pdf   -> %d %s, %d bytes"
          % (p.status_code, p.headers.get("content-type"), len(p.content)))
    if p.status_code != 200:
        print("FAIL: /pdf returned %d" % p.status_code)
        return 1
    if not p.content.startswith(b"%PDF"):
        print("FAIL: /pdf body is not a PDF (starts %r)" % p.content[:16])
        return 1

    import io
    import pypdfium2
    pages = len(pypdfium2.PdfDocument(io.BytesIO(p.content)))
    print("pdf pages -> %d" % pages)
    if pages < 1:
        print("FAIL: compiled PDF has %d pages" % pages)
        return 1

    # The left pane.
    x = client.get("/latex/%s" % job_id)
    print("GET /latex -> %d, %d chars" % (x.status_code, len(x.text)))
    if x.status_code != 200 or "\\begin{document}" not in x.text:
        print("FAIL: /latex did not return a LaTeX document")
        return 1
    return 0


def main() -> int:
    conv = serve.materialize_fp32_graphs()
    print("materialised %.1f MB" % conv["materialized_mb"])
    for line in conv["converted"]:
        print("   ", line)
    if conv["skipped"]:
        print("FAIL: graphs skipped during back-conversion:", conv["skipped"])
        return 1

    serve._make_workers_persistent()
    warm = serve.warm_up()
    print("warm-up:", warm)

    if warm["warmup_chars"] < MIN_CHARS:
        print("FAIL: warm-up produced %d chars, expected at least %d"
              % (warm["warmup_chars"], MIN_CHARS))
        return 1

    # The UI is served from this same app, so a broken or missing page should
    # fail the build too. TestClient is NOT used as a context manager on
    # purpose: that would re-run the startup event and reload every model.
    from fastapi.testclient import TestClient

    serve._ready = True          # startup already ran, in-process, above
    client = TestClient(serve.app)

    rc = check_ui(client)
    if rc:
        return rc

    r = client.get("/health")
    print("GET /health-> %d" % r.status_code)
    if r.status_code != 200:
        print("FAIL: /health returned %d" % r.status_code)
        return 1

    r = client.get("/progress")
    print("GET /progress -> %d %s" % (r.status_code, r.json()))
    if r.status_code != 200:
        print("FAIL: /progress returned %d" % r.status_code)
        return 1

    rc = check_job(client)
    if rc:
        return rc

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
