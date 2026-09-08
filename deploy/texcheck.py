"""
Compile real generated documents with the container's TeX, in the build.

The web UI's whole right pane is a PDF: app.py runs the pipeline, then shells
out to xelatex/pdflatex on the main.tex the pipeline wrote. A TeX install that
is merely *present* is not enough -- a missing package makes the compile fail,
and a missing CJK font makes it SUCCEED while dropping every Chinese glyph on
the floor. Both must fail the build, not ship.

The fixtures under deploy/texcheck/ are real pipeline output, not hand-written
minimal examples, so they exercise the actual preamble
pipeline/latex_builder.py emits:

    latin/  \\usepackage[utf8]{inputenc} + paracol   -> pdflatex, the
            visual-fidelity two-column path (PRISM_VISUAL_FIDELITY=1, which
            app.py:64 sets for every web job)
    cjk/    \\usepackage{xeCJK}          + paracol   -> xelatex, CJK glyphs

Compiler selection below is copied from app.py:85 deliberately: if that rule
and this one ever disagree, this check is testing the wrong binary.

    python deploy/texcheck.py
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "texcheck"

# Any of these in the log means the PDF is wrong even if TeX exited 0.
FATAL_LOG = [
    # A glyph the font could not draw. This is the "PDF with holes" case: with
    # no CJK font, xeCJK silently renders CJK with the Latin font and every
    # ideograph becomes one of these lines instead of ink.
    (r"Missing character", "a glyph was dropped from the PDF"),
    (r"No CJK font family", "xeCJK has no CJK font configured"),
    (r"^! LaTeX Error", "LaTeX error"),
    (r"^! Package .* Error", "package error"),
    (r"file .* not found", "a .sty/.cls is missing from the TeX install"),
    (r"^! Font .* not (loadable|found)", "a font is missing"),
]

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _pdf_pages_and_text(pdf: Path):
    import pypdfium2
    doc = pypdfium2.PdfDocument(str(pdf))
    text = "".join(p.get_textpage().get_text_range() for p in doc)
    return len(doc), text


def check(name: str, min_cjk_glyphs: int = 0) -> list[str]:
    src = FIXTURES / name
    tex = src / "main.tex"
    if not tex.exists():
        return [f"{name}: fixture missing at {tex}"]

    body = tex.read_text(encoding="utf-8")
    # app.py:85, verbatim.
    compiler = "xelatex" if "\\usepackage{xeCJK}" in body else "pdflatex"

    work = Path(tempfile.mkdtemp(prefix=f"texcheck_{name}_"))
    try:
        shutil.copytree(src, work / "doc")
        doc = work / "doc"
        proc = subprocess.run(
            [compiler, "-interaction=nonstopmode", "main.tex"],
            cwd=str(doc), capture_output=True, text=True, timeout=300,
        )
        pdf = doc / "main.pdf"
        log = (doc / "main.log")
        log_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""

        pkgs = sorted(set(re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]*)\}", body)))
        print(f"  {name}: {compiler}, packages {', '.join(pkgs)}")

        errs = []
        if proc.returncode != 0:
            tail = "\n".join(log_text.splitlines()[-25:]) or proc.stdout[-1500:]
            errs.append(f"{name}: {compiler} exited {proc.returncode}\n{tail}")
        if not pdf.exists():
            errs.append(f"{name}: no main.pdf produced")
            return errs

        pages, text = _pdf_pages_and_text(pdf)
        print(f"  {name}: {pdf.stat().st_size} bytes, {pages} page(s)")
        if pages < 1:
            errs.append(f"{name}: PDF has {pages} pages")

        for pat, why in FATAL_LOG:
            hits = re.findall(pat + r".*", log_text, re.M)
            if hits:
                errs.append(f"{name}: {why} -- {len(hits)} occurrence(s), "
                            f"first: {hits[0].strip()[:180]}")

        if min_cjk_glyphs:
            # The log check above catches dropped glyphs, but only if the font
            # machinery reported them. Read the ink back out of the PDF too.
            found = len(CJK_RE.findall(text))
            print(f"  {name}: {found} CJK glyphs recovered from the PDF text")
            if found < min_cjk_glyphs:
                errs.append(f"{name}: expected at least {min_cjk_glyphs} CJK "
                            f"glyphs in the PDF, found {found}")
        return errs
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    for exe in ("xelatex", "pdflatex"):
        path = shutil.which(exe)
        print(f"{exe}: {path or 'NOT FOUND'}")
        if not path:
            print(f"FAIL: {exe} is not on PATH; app.py:85 shells out to it by name")
            return 1

    errs = check("latin") + check("cjk", min_cjk_glyphs=100)
    if errs:
        print("\nTEX CHECK FAILED")
        for e in errs:
            print(" *", e)
        return 1

    print("TEX CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
