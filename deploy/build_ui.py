"""
Build deploy/ui.html from web/index.html.

The container serves a markdown API, not the LaTeX/PDF job queue app.py drives,
so the page keeps its markup and all ~10 KB of its CSS and swaps only the parts
that are actually different:

  * the result view becomes one pane with a Rendered / Markdown toggle,
    a copy button and a .md download, instead of a LaTeX-source + PDF-iframe
    split (the container has no TeX distribution);
  * the five-step progress list is replaced by real state. The original drove
    those steps off hard-coded timers (STEP_MS = [500, 9000, 20000, 32000,
    45000]) with no connection to the pipeline; this polls /progress and shows
    queue position, elapsed seconds, and page N of M for PDFs;
  * the client rejects >30 MB before uploading, and treats a suspiciously short
    response as an error rather than rendering a blank page.

    python deploy/build_ui.py        # -> deploy/ui.html
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "web" / "index.html"
DST = ROOT / "deploy" / "ui.html"

# KaTeX renders the maths, marked renders the markdown. Both come from the same
# CDN the page already uses for highlight.js. They are third-party requests, so
# they do not consume this service's own request concurrency.
HEAD_ADD = """  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css" />
  <script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
  <script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
"""

STATUS_BLOCK = """      <div class="steps" id="steps">
        <div class="step active"><div class="step-node"></div><span id="proc-status">Starting…</span></div>
      </div>
"""

RESULT_PANE = """    <div class="pane" style="flex:1; min-height:0;">
      <div class="pane-bar">
        <span class="pane-label" id="pane-label">Rendered</span>
        <button class="btn btn-outline btn-sm" id="toggle-btn" onclick="toggleView()">
          View markdown
        </button>
        <button class="btn btn-outline btn-sm" onclick="copyMarkdown()">
          <svg viewBox="0 0 20 20" fill="currentColor">
            <path d="M8 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z"/>
            <path d="M6 3a2 2 0 00-2 2v11a2 2 0 002 2h8a2 2 0 002-2V5a2 2 0 00-2-2 3 3 0 01-3 3H9a3 3 0 01-3-3z"/>
          </svg>
          Copy
        </button>
      </div>
      <div class="code-scroll" id="render-scroll">
        <div id="rendered-md" class="rendered-md"></div>
      </div>
      <div class="code-scroll" id="raw-scroll" style="display:none;">
        <pre id="md-pre"><code id="md-code"></code></pre>
      </div>
    </div>

  </div>

"""

EXTRA_CSS = """
    /* Rendered markdown. .code-scroll paints var(--code-bg) (near-black) for
       the highlight.js block, so the rendered pane must override it or dark
       body text lands on a dark ground. Variable names are the page's own:
       --text / --text-3 / --border / --surface, NOT --text-1 or --line. */
    #render-scroll { background: var(--surface); }
    .rendered-md { padding: 24px 28px; color: var(--text); line-height: 1.68;
                   font-size: 14.5px; text-align: left; }
    .rendered-md h1, .rendered-md h2, .rendered-md h3,
    .rendered-md h4 { margin: 1.3em 0 .5em; line-height: 1.3; color: var(--text); }
    .rendered-md h1 { font-size: 1.5em; }
    .rendered-md h2 { font-size: 1.26em; }
    .rendered-md h3 { font-size: 1.1em; }
    .rendered-md h4 { font-size: 1em; }
    .rendered-md p, .rendered-md li { margin: .65em 0; }
    .rendered-md ul, .rendered-md ol { padding-left: 1.4em; }
    .rendered-md table { border-collapse: collapse; margin: 1.1em 0;
                   max-width: 100%; display: block; overflow-x: auto;
                   font-size: .94em; }
    .rendered-md th, .rendered-md td { border: 1px solid var(--border);
                   padding: 6px 11px; text-align: left; vertical-align: top; }
    .rendered-md th { background: var(--surface-2); font-weight: 700; }
    .rendered-md img { max-width: 100%; }
    .rendered-md code { font-family: var(--mono); font-size: .9em;
                   background: var(--surface-2); padding: 1px 5px;
                   border-radius: 4px; }
    .rendered-md pre code { display: block; padding: 12px 14px; overflow-x: auto; }
    .rendered-md blockquote { margin: .8em 0; padding-left: 1em;
                   border-left: 3px solid var(--border); color: var(--text-2); }
    .rendered-md .katex { font-size: 1.02em; }
    .rendered-md .katex-display { overflow-x: auto; overflow-y: hidden;
                   padding: 4px 0; margin: 1em 0; }
    .page-rule { margin: 2em 0 .6em; border: 0; border-top: 1px dashed var(--border); }
    .page-rule-label { font-size: 11px; letter-spacing: .09em;
                   text-transform: uppercase; color: var(--text-3);
                   font-weight: 700; margin: 0 0 1em; }

    /* Raw markdown pane: same treatment #latex-pre had. */
    #md-pre {
      margin: 0; padding: 20px 24px;
      font-family: var(--mono); font-size: 12.5px; line-height: 1.72;
      white-space: pre-wrap; word-break: break-word; min-height: 100%;
      color: #abb2bf; text-align: left;
    }
"""

JS = r"""<script>
  const uploadPanel     = document.getElementById('upload-panel');
  const processingPanel = document.getElementById('processing-panel');
  const resultPanel     = document.getElementById('result-panel');
  const errorPanel      = document.getElementById('error-panel');

  const dropZone    = document.getElementById('drop-zone');
  const fileInput   = document.getElementById('file-input');
  const procLabel   = document.getElementById('proc-label');
  const procElapsed = document.getElementById('proc-elapsed');
  const procStatus  = document.getElementById('proc-status');

  const resultThumb = document.getElementById('result-thumb');
  const resultName  = document.getElementById('result-name');
  const downloadBtn = document.getElementById('download-btn');
  const renderedMd  = document.getElementById('rendered-md');
  const mdCode      = document.getElementById('md-code');
  const renderScroll= document.getElementById('render-scroll');
  const rawScroll   = document.getElementById('raw-scroll');
  const paneLabel   = document.getElementById('pane-label');
  const toggleBtn   = document.getElementById('toggle-btn');
  const errorMsg    = document.getElementById('error-msg');
  const toast       = document.getElementById('toast');

  // Cloud Run hard-caps request bodies at 32 MB; refuse before uploading so a
  // large file fails instantly and clearly rather than after a long upload.
  const MAX_BYTES = 30 * 1024 * 1024;
  // Two of the three deployment failures produced empty or near-empty output
  // instead of an error. Treat a suspiciously short result as a failure.
  const MIN_CHARS = 200;
  const OK_EXT = ['png','jpg','jpeg','webp','bmp','tif','tiff','pdf'];

  let elapsedTimer = null, pollTimer = null, startTime = null;
  let previewUrl = null, mdBlobUrl = null, mdSrc = '', showingRaw = false;

  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', e => { if (!dropZone.contains(e.relatedTarget)) dropZone.classList.remove('drag-over'); });
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0]; if (f) submit(f);
  });
  fileInput.addEventListener('change', () => { if (fileInput.files[0]) submit(fileInput.files[0]); });

  function show(id) {
    uploadPanel.style.display     = id === 'upload'     ? 'flex' : 'none';
    processingPanel.style.display = id === 'processing' ? 'flex' : 'none';
    resultPanel.style.display     = id === 'result'     ? 'flex' : 'none';
    errorPanel.style.display      = id === 'error'      ? 'flex' : 'none';
  }

  function fail(msg) {
    stopTimers();
    errorMsg.textContent = msg;
    show('error');
  }

  function stopTimers() {
    if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
    if (pollTimer)    { clearInterval(pollTimer);    pollTimer = null; }
  }

  function fmt(s) {
    const m = Math.floor(s / 60), r = Math.floor(s % 60);
    return m ? `${m}m ${String(r).padStart(2,'0')}s` : `${r}s`;
  }

  async function submit(file) {
    const ext = (file.name.split('.').pop() || '').toLowerCase();
    if (!OK_EXT.includes(ext)) {
      fail(`Unsupported file type ".${ext}".\n\nSend a PDF or an image (${OK_EXT.filter(e=>e!=='pdf').join(', ')}).`);
      return;
    }
    if (file.size > MAX_BYTES) {
      const mb = (file.size / 1048576).toFixed(1);
      fail(`That file is ${mb} MB, over the 30 MB limit.\n\n` +
           `Cloud Run caps request bodies at 32 MB, so it is rejected here ` +
           `rather than failing part-way through the upload.`);
      return;
    }

    resultName.textContent = file.name;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (file.type.startsWith('image/')) {
      previewUrl = URL.createObjectURL(file);
      resultThumb.src = previewUrl;
      resultThumb.style.display = '';
    } else {
      resultThumb.style.display = 'none';
    }

    show('processing');
    procLabel.textContent = 'Reading your document…';
    procStatus.textContent = 'Uploading…';
    startTime = Date.now();
    procElapsed.textContent = '0s';
    elapsedTimer = setInterval(() => {
      procElapsed.textContent = fmt((Date.now() - startTime) / 1000);
    }, 250);
    pollTimer = setInterval(pollProgress, 1500);
    pollProgress();

    const fd = new FormData();
    fd.append('file', file, file.name);

    let res;
    try {
      res = await fetch('/parse', { method: 'POST', body: fd });
    } catch (e) {
      fail('Could not reach the server.\n\n' + e);
      return;
    }

    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const j = await res.json();
        if (j && j.detail) detail = j.detail;
      } catch (_) {
        try { detail = (await res.text()).slice(0, 800) || detail; } catch (_) {}
      }
      fail(detail);
      return;
    }

    const text = await res.text();
    stopTimers();

    // Loudly refuse to render a blank page.
    if (!text || text.trim().length < MIN_CHARS) {
      fail(`The pipeline returned only ${text.trim().length} characters, which ` +
           `almost certainly means it failed rather than that the page was ` +
           `empty.\n\nNothing has been rendered. Try another file, or check ` +
           `the service logs.`);
      return;
    }

    mdSrc = text;
    renderMarkdown(text);
    mdCode.textContent = text;
    if (window.hljs) { try { hljs.highlightElement(mdCode); } catch (_) {} }

    if (mdBlobUrl) URL.revokeObjectURL(mdBlobUrl);
    mdBlobUrl = URL.createObjectURL(new Blob([text], { type: 'text/markdown' }));
    downloadBtn.href = mdBlobUrl;
    downloadBtn.download = file.name.replace(/\.[^.]+$/, '') + '.md';

    showingRaw = false;
    applyView();
    show('result');
  }

  function renderMarkdown(text) {
    // Page separators the server emits for multi-page PDFs.
    let src = text.replace(/<!--\s*page (\d+)\s*-->/g,
      (_, n) => (n === '1' ? '' : '\n\n<hr class="page-rule">\n\n') +
                `<div class="page-rule-label">Page ${n}</div>\n\n`);

    // Stash maths BEFORE markdown parsing and put it back after.
    // marked eats the backslashes in \[ ... \] and \( ... \), so KaTeX never
    // sees the delimiter and display equations render as literal LaTeX. Inline
    // $...$ survives, which is why only display maths looked broken.
    // Longest delimiters first, or $$ would be matched as two empty $...$.
    const math = [];
    const stash = m => `@@PRISMMATH${math.push(m) - 1}@@`;
    src = src.replace(/\$\$[\s\S]+?\$\$/g, stash)
             .replace(/\\\[[\s\S]+?\\\]/g, stash)
             .replace(/\\\([\s\S]+?\\\)/g, stash)
             .replace(/\$[^$\n]+?\$/g, stash);

    let html;
    try {
      html = window.marked ? marked.parse(src, { breaks: false, gfm: true }) : src;
    } catch (e) {
      html = src;
    }
    html = html.replace(/@@PRISMMATH(\d+)@@/g, (_, i) => math[+i]);
    renderedMd.innerHTML = html;
    if (window.renderMathInElement) {
      try {
        renderMathInElement(renderedMd, {
          delimiters: [
            { left: '$$', right: '$$', display: true  },
            { left: '\\[', right: '\\]', display: true  },
            { left: '$',  right: '$',  display: false },
            { left: '\\(', right: '\\)', display: false }
          ],
          throwOnError: false
        });
      } catch (_) {}
    }
  }

  function applyView() {
    renderScroll.style.display = showingRaw ? 'none' : '';
    rawScroll.style.display    = showingRaw ? '' : 'none';
    paneLabel.textContent      = showingRaw ? 'Markdown' : 'Rendered';
    toggleBtn.textContent      = showingRaw ? 'View rendered' : 'View markdown';
  }

  function toggleView() { showingRaw = !showingRaw; applyView(); }

  async function pollProgress() {
    try {
      const r = await fetch('/progress', { cache: 'no-store' });
      if (!r.ok) return;
      const p = await r.json();
      if (!p.ready) {
        procStatus.textContent = 'Server still loading its models…';
        return;
      }
      if (p.waiting > 0 && p.state === 'running') {
        // Another parse holds the semaphore; this one has not started yet.
        procStatus.textContent =
          `Queued behind ${p.waiting} job${p.waiting > 1 ? 's' : ''} — ` +
          `one page is processed at a time`;
        procLabel.textContent = 'Waiting for the server…';
      } else if (p.state === 'running') {
        procLabel.textContent = 'Reading your document…';
        procStatus.textContent = p.pages > 1
          ? `Processing page ${p.page} of ${p.pages}`
          : 'Processing';
      } else {
        procStatus.textContent = 'Starting…';
      }
    } catch (_) { /* keep the elapsed clock running */ }
  }

  function copyMarkdown() {
    navigator.clipboard.writeText(mdSrc).then(() => {
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 1600);
    });
  }

  function reset() {
    stopTimers();
    fileInput.value = '';
    show('upload');
  }

  show('upload');
</script>
"""


def build() -> None:
    html = SRC.read_text(encoding="utf-8")

    # 1. head: KaTeX + marked
    anchor = '  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>\n'
    assert anchor in html, "highlight.js script tag not found"
    html = html.replace(anchor, anchor + HEAD_ADD, 1)

    # 2. extra CSS for rendered markdown, appended to the existing <style>
    i = html.rindex("</style>")
    html = html[:i] + EXTRA_CSS + html[i:]

    # 3. accept PDFs
    html = html.replace('accept="image/*"', 'accept=".png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff,.pdf"')
    html = html.replace('<span class="chip">TIFF</span>',
                        '<span class="chip">TIFF</span>\n        <span class="chip">PDF</span>')
    html = html.replace("<p class=\"drop-title\">Drop a document image here</p>",
                        "<p class=\"drop-title\">Drop a document or PDF here</p>")
    html = html.replace("and give you clean LaTeX + PDF.",
                        "and give you clean Markdown.")
    html = html.replace('<span class="wordmark-tag">Document → LaTeX</span>',
                        '<span class="wordmark-tag">Document → Markdown</span>')
    html = html.replace("<title>PRISM — Document to LaTeX</title>",
                        "<title>PRISM — Document to Markdown</title>")

    # 4. honest progress instead of the timer-driven step list
    a = html.index('      <div class="steps" id="steps">')
    b = html.index("\n      </div>", a) + len("\n      </div>\n")
    html = html[:a] + STATUS_BLOCK + html[b:]

    # 5. result view: one pane with a Rendered / Markdown toggle
    html = html.replace('download="output.pdf"', 'download="output.md"')
    html = html.replace("        Download PDF\n", "        Download .md\n")
    a = html.index('    <div class="split">')
    b = html.index("  <!-- Error -->")
    html = html[:a] + RESULT_PANE + html[b:]

    # 6. new client logic
    a = html.index("<script>\n  const uploadPanel")
    b = html.index("</script>", a) + len("</script>\n")
    html = html[:a] + JS + html[b:]

    DST.write_text(html, encoding="utf-8")
    print(f"wrote {DST.relative_to(ROOT)}  ({len(html)} bytes)")
    for must in ("id=\"rendered-md\"", "toggleView", "/progress", "MAX_BYTES",
                 "katex", "marked.min.js"):
        assert must in html, f"missing {must}"
    # the #pdf-viewer CSS rule may remain; the ELEMENT must not
    assert '<iframe id="pdf-viewer"' not in html, "PDF iframe still present"
    assert "STEP_MS" not in html, "timer-driven steps still present"
    print("checks passed")


if __name__ == "__main__":
    build()
