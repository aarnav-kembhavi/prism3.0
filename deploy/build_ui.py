"""
Build deploy/ui.html from web/index.html.

The ORIGINAL layout is preserved: `<div class="split">` with two `<div
class="pane">` children, each with its own `.pane-bar`. Those wrappers, and all
~10 KB of the page's CSS, are copied through byte for byte. Only the CONTENTS
of the two panes change, because neither original pane can work in this
container:

    original (web/index.html @ 2e62b83)   ->  here
    left  : LaTeX Source, <pre id=...>    ->  Source Page, rasterised page image
    right : PDF Preview,  <iframe>        ->  Markdown, rendered output

Why the contents had to change, rather than a preference:

  * the right pane's iframe is fed by /pdf/{job_id}, which compiles LaTeX with
    xelatex/pdflatex. This image carries no TeX distribution, and adding
    TeXLive would put 1-2 GB onto an image whose cold start is a tracked
    number. The pane shows the markdown /parse returns instead.
  * the left pane shows the source page, as the first version of this UI did
    (web/index.html @ a7a367b, `<img id="input-preview">`). For an image upload
    the browser renders the file it already holds; a PDF has to come back from
    the server's rasteriser via GET /page/{token}/{n}.

Everything else kept from the previous iteration is additive and does not touch
layout: the /progress-driven queue + elapsed + page-N-of-M status, the 30 MB
client-side check, the under-200-character error, and CSS that names variables
this palette actually defines.

    python deploy/build_ui.py        # -> deploy/ui.html
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "web" / "index.html"
DST = ROOT / "deploy" / "ui.html"

# KaTeX renders the maths, marked renders the markdown, from the same CDN the
# page already uses for highlight.js. Third-party requests, so they do not
# consume this service's own request concurrency.
HEAD_ADD = """  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css" />
  <script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
  <script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
"""

# The original animated five steps off hard-coded timers
# (STEP_MS = [500, 9000, 20000, 32000, 45000]) with no connection to the
# pipeline. Same markup, one row, driven by /progress.
STATUS_BLOCK = """      <div class="steps" id="steps">
        <div class="step active"><div class="step-node"></div><span id="proc-status">Starting…</span></div>
      </div>
"""

LEFT_PANE_BODY = """        <div class="page-scroll" id="page-scroll">
          <div id="page-stack"></div>
        </div>
"""

RIGHT_PANE_BODY = """        <div class="code-scroll md-scroll" id="render-scroll">
          <div id="rendered-md" class="rendered-md"></div>
        </div>
"""

EXTRA_CSS = """
    /* Source-page pane: A4 sheets on a recessed ground. */
    .page-scroll { flex: 1; overflow: auto; background: var(--surface-2);
                   padding: 18px; min-height: 0; }
    .page-sheet { position: relative; background: #fff; margin: 0 auto 16px;
                  width: 100%; max-width: 560px; aspect-ratio: 1 / 1.4142;
                  border: 1px solid var(--border); box-shadow: var(--shadow);
                  display: flex; align-items: center; justify-content: center;
                  overflow: hidden; }
    .page-sheet img { width: 100%; height: 100%; object-fit: contain;
                      display: block; }
    .page-missing { color: var(--text-3); font-size: 12.5px; text-align: center;
                    padding: 0 20px; }
    .page-num { text-align: center; font-size: 11px; letter-spacing: .08em;
                text-transform: uppercase; color: var(--text-3);
                font-weight: 700; margin: -8px 0 18px; }

    /* Rendered markdown. .code-scroll paints var(--code-bg) (near-black) for
       the highlight.js block, so this pane must override it or dark body text
       lands on a dark ground. Variable names are the page's own --
       --text / --text-3 / --border / --surface, NOT --text-1 or --line. */
    .md-scroll { background: var(--surface); }
    .rendered-md { padding: 22px 26px; color: var(--text); line-height: 1.68;
                   font-size: 14px; text-align: left; }
    .rendered-md h1, .rendered-md h2, .rendered-md h3,
    .rendered-md h4 { margin: 1.3em 0 .5em; line-height: 1.3; color: var(--text); }
    .rendered-md h1 { font-size: 1.45em; }
    .rendered-md h2 { font-size: 1.22em; }
    .rendered-md h3 { font-size: 1.08em; }
    .rendered-md h4 { font-size: 1em; }
    .rendered-md p, .rendered-md li { margin: .6em 0; }
    .rendered-md ul, .rendered-md ol { padding-left: 1.35em; }
    .rendered-md table { border-collapse: collapse; margin: 1em 0;
                   max-width: 100%; display: block; overflow-x: auto;
                   font-size: .93em; }
    .rendered-md th, .rendered-md td { border: 1px solid var(--border);
                   padding: 5px 10px; text-align: left; vertical-align: top; }
    .rendered-md th { background: var(--surface-2); font-weight: 700; }
    .rendered-md img { max-width: 100%; }
    .rendered-md code { font-family: var(--mono); font-size: .9em;
                   background: var(--surface-2); padding: 1px 5px;
                   border-radius: 4px; }
    .rendered-md pre code { display: block; padding: 12px 14px; overflow-x: auto; }
    .rendered-md blockquote { margin: .8em 0; padding-left: 1em;
                   border-left: 3px solid var(--border); color: var(--text-2); }
    .rendered-md .katex { font-size: 1.01em; }
    .rendered-md .katex-display { overflow-x: auto; overflow-y: hidden;
                   padding: 4px 0; margin: 1em 0; }
    .page-rule { margin: 1.8em 0 .5em; border: 0;
                 border-top: 1px dashed var(--border); }
    .page-rule-label { font-size: 11px; letter-spacing: .09em;
                   text-transform: uppercase; color: var(--text-3);
                   font-weight: 700; margin: 0 0 1em; }
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
  const pageStack   = document.getElementById('page-stack');
  const renderedMd  = document.getElementById('rendered-md');
  const errorMsg    = document.getElementById('error-msg');
  const toast       = document.getElementById('toast');

  // Cloud Run hard-caps request bodies at 32 MB; refuse before uploading so a
  // large file fails instantly rather than part-way through the upload.
  const MAX_BYTES = 30 * 1024 * 1024;
  // Two of the three deployment failures produced empty or near-empty output
  // instead of an error. Treat a suspiciously short result as a failure.
  const MIN_CHARS = 200;
  const OK_EXT = ['png','jpg','jpeg','webp','bmp','tif','tiff','pdf'];

  let elapsedTimer = null, pollTimer = null, startTime = null;
  let previewUrl = null, mdBlobUrl = null, mdSrc = '';

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

  function fail(msg) { stopTimers(); errorMsg.textContent = msg; show('error'); }

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
      fail(`That file is ${(file.size/1048576).toFixed(1)} MB, over the 30 MB limit.\n\n` +
           `Cloud Run caps request bodies at 32 MB, so it is rejected here ` +
           `rather than failing part-way through the upload.`);
      return;
    }

    resultName.textContent = file.name;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
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

    const token = res.headers.get('X-Prism-Doc');
    const nPages = parseInt(res.headers.get('X-Prism-Pages') || '1', 10) || 1;
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

    buildPages(token, nPages);
    mdSrc = text;
    renderMarkdown(text);

    if (mdBlobUrl) URL.revokeObjectURL(mdBlobUrl);
    mdBlobUrl = URL.createObjectURL(new Blob([text], { type: 'text/markdown' }));
    downloadBtn.href = mdBlobUrl;
    downloadBtn.download = file.name.replace(/\.[^.]+$/, '') + '.md';

    show('result');
  }

  // Left pane: the source page(s), as A4 sheets. A PDF's pages come back from
  // the server's rasteriser; an image is the local file we already have.
  function buildPages(token, nPages) {
    pageStack.innerHTML = '';
    for (let i = 0; i < nPages; i++) {
      const sheet = document.createElement('div');
      sheet.className = 'page-sheet';
      const img = document.createElement('img');
      img.alt = `Source page ${i + 1}`;
      img.loading = i > 1 ? 'lazy' : 'eager';
      img.src = token ? `/page/${token}/${i}` : (previewUrl || '');
      img.addEventListener('error', () => {
        if (previewUrl && img.src !== previewUrl) { img.src = previewUrl; return; }
        // No local copy to fall back on (a PDF), and the server could not
        // return the page. Say so rather than leaving a broken-image icon.
        img.remove();
        const note = document.createElement('div');
        note.className = 'page-missing';
        note.textContent = 'Source page unavailable';
        sheet.appendChild(note);
      }, { once: true });
      sheet.appendChild(img);
      pageStack.appendChild(sheet);
      if (nPages > 1) {
        const cap = document.createElement('div');
        cap.className = 'page-num';
        cap.textContent = `Page ${i + 1} of ${nPages}`;
        pageStack.appendChild(cap);
      }
    }
  }

  function renderMarkdown(text) {
    let src = text.replace(/<!--\s*page (\d+)\s*-->/g,
      (_, n) => (n === '1' ? '' : '\n\n<hr class="page-rule">\n\n') +
                `<div class="page-rule-label">Page ${n}</div>\n\n`);

    // Stash maths BEFORE markdown parsing and put it back after. marked eats
    // the backslashes in \[ ... \] and \( ... \), so KaTeX never sees the
    // delimiter and display equations render as literal LaTeX; inline $...$
    // survives, which is why only display maths looked broken. Longest
    // delimiters first, or $$ is matched as two empty $...$.
    const math = [];
    const stash = m => `@@PRISMMATH${math.push(m) - 1}@@`;
    src = src.replace(/\$\$[\s\S]+?\$\$/g, stash)
             .replace(/\\\[[\s\S]+?\\\]/g, stash)
             .replace(/\\\([\s\S]+?\\\)/g, stash)
             .replace(/\$[^$\n]+?\$/g, stash);

    let html;
    try {
      html = window.marked ? marked.parse(src, { breaks: false, gfm: true }) : src;
    } catch (e) { html = src; }
    html = html.replace(/@@PRISMMATH(\d+)@@/g, (_, i) => math[+i]);
    renderedMd.innerHTML = html;

    if (window.renderMathInElement) {
      try {
        renderMathInElement(renderedMd, {
          delimiters: [
            { left: '$$',  right: '$$',  display: true  },
            { left: '\\[', right: '\\]', display: true  },
            { left: '$',   right: '$',   display: false },
            { left: '\\(', right: '\\)', display: false }
          ],
          throwOnError: false
        });
      } catch (_) {}
    }
  }

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
        procLabel.textContent = 'Waiting for the server…';
        procStatus.textContent =
          `Queued behind ${p.waiting} job${p.waiting > 1 ? 's' : ''} — ` +
          `one document is processed at a time`;
      } else if (p.state === 'running') {
        procLabel.textContent = 'Reading your document…';
        procStatus.textContent = p.pages > 1
          ? `Processing page ${p.page} of ${p.pages}`
          : 'Processing';
      } else {
        // state 'idle' here does not mean nothing is happening: /progress is
        // per-instance, and a poll can land on an instance that is not the one
        // running this parse. Our own request is demonstrably in flight, so
        // say that rather than something false.
        procStatus.textContent = 'Processing';
      }
    } catch (_) { /* keep the elapsed clock running */ }
  }

  function copyMarkdown() {
    navigator.clipboard.writeText(mdSrc).then(() => {
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 1600);
    });
  }

  function reset() { stopTimers(); fileInput.value = ''; show('upload'); }

  show('upload');
</script>
"""


def build() -> None:
    html = SRC.read_text(encoding="utf-8")
    orig = html

    # 1. head: KaTeX + marked
    anchor = '  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>\n'
    assert anchor in html, "highlight.js script tag not found"
    html = html.replace(anchor, anchor + HEAD_ADD, 1)

    # 2. extra CSS appended to the page's own <style>
    i = html.rindex("</style>")
    html = html[:i] + EXTRA_CSS + html[i:]

    # 3. accept PDFs as well as images
    html = html.replace('accept="image/*"',
                        'accept=".png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff,.pdf"')
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

    # 4. real progress state in place of the timer-driven step list
    a = html.index('      <div class="steps" id="steps">')
    b = html.index("\n      </div>", a) + len("\n      </div>\n")
    html = html[:a] + STATUS_BLOCK + html[b:]

    # 5. THE TWO PANES. <div class="split">, both <div class="pane"> and both
    #    <div class="pane-bar"> wrappers are left exactly as they are; only the
    #    labels and the pane bodies change.
    #
    #    Lift the Copy button out of the source pane rather than retyping its
    #    markup, so the SVG and classes stay byte-identical, and move it to the
    #    markdown pane where there is now something to copy.
    m = re.search(r'\n(\s*)<button class="btn btn-outline btn-sm" onclick="copyLatex\(\)">'
                  r'.*?</button>\n', html, re.S)
    assert m, "original Copy button not found"
    copy_btn = m.group(0).replace("copyLatex()", "copyMarkdown()")
    html = html[:m.start()] + "\n" + html[m.end():]

    html = html.replace('<span class="pane-label">LaTeX Source</span>',
                        '<span class="pane-label">Source Page</span>', 1)
    html = html.replace('<span class="pane-label">PDF Preview</span>',
                        '<span class="pane-label">Markdown</span>'
                        + copy_btn.rstrip("\n"), 1)

    # left pane body: page sheets instead of the LaTeX <pre>
    m = re.search(r'[ \t]*<div class="code-scroll">\s*<pre id="latex-pre">'
                  r'<code id="latex-code"></code></pre>\s*</div>\n', html)
    assert m, "original left-pane code-scroll not found"
    html = html[:m.start()] + LEFT_PANE_BODY + html[m.end():]

    # right pane body: rendered markdown instead of the PDF iframe
    m = re.search(r'[ \t]*<iframe id="pdf-viewer" title="PDF output"></iframe>\n', html)
    assert m, "original PDF iframe not found"
    html = html[:m.start()] + RIGHT_PANE_BODY + html[m.end():]

    # 6. the download is markdown now, not a compiled PDF
    html = html.replace('download="output.pdf"', 'download="output.md"')
    html = html.replace("        Download PDF\n", "        Download .md\n")

    # 7. new client logic
    a = html.index("<script>\n  const uploadPanel")
    b = html.index("</script>", a) + len("</script>\n")
    html = html[:a] + JS + html[b:]

    DST.write_text(html, encoding="utf-8")
    print(f"wrote {DST.relative_to(ROOT)}  ({len(html)} bytes, original {len(orig)})")

    # The layout is the point: assert the original two-pane scaffolding survived.
    assert html.count('<div class="split">') == 1, "split container lost"
    assert html.count('<div class="pane">') == 2, "expected exactly two panes"
    assert html.count('<div class="pane-bar">') == 2, "pane bars lost"
    for must in ('id="page-stack"', 'id="rendered-md"', '/page/', '/progress',
                 "MAX_BYTES", "katex", "marked.min.js",
                 '<span class="pane-label">Source Page</span>',
                 '<span class="pane-label">Markdown</span>'):
        assert must in html, f"missing {must}"
    assert '<iframe id="pdf-viewer"' not in html, "PDF iframe still present"
    assert "STEP_MS" not in html, "timer-driven steps still present"
    assert "copyLatex" not in html, "stale copyLatex handler"
    print("checks passed: split + 2 panes + 2 pane-bars intact")


if __name__ == "__main__":
    build()
