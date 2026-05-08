"""
latex_builder.py
----------------
Maps YOLO class names to LaTeX wrappers and assembles the document.
This is the ONLY file that knows about LaTeX syntax.

Fixes applied:
  B — header_logo placed above paracol block, right-aligned, not mid-column
  C — Section-header maps to \\subsection*{} not \\section{} (IEEE style)
      Title maps to \\begin{center}\\textbf{\\large ...}\\end{center}
  E — list items: each \\item on its own line; itemize grouping preserved
  F — soft-hyphen artifact cleaner strips "word- suffix" OCR artifacts
      thousands-separator dot confusion fixed: "7.352" → "7,352"
"""

import re
from typing import List, Optional
import os


# ----------------------------------------------------------------
# Post-processing: clean OCR output before wrapping in LaTeX
# ----------------------------------------------------------------

# Fix 4 — Soft-hyphen column-break artifact with optional spaces around hyphen:
#   "pa- rameters" → "parameters"
#   "con - taining" → "containing"   (OCR inserts spaces around the hyphen)
# Fires when: word-char, optional space, hyphen, optional space(s), lowercase.
_SOFT_HYPHEN_RE = re.compile(r'(\w) ?- +([a-z])')

# Fix 7 — Thousands-separator dot confusion: "7.352" → "7,352"
# Broadened to also catch trailing non-digit chars like capital-O OCR misread:
# "209.00O" → tries digits then falls through; handled by the O→0 pass first.
_THOUSANDS_DOT_RE = re.compile(r'\b(\d{1,3})\.(\d{3})\b')

# Fix 5 — Common OCR character confusions (applied before escaping)
_OCR_FIXES = [
    # "t0" at word boundary where 0 is misread "o" → "to"
    (re.compile(r'\bt0\b'), 'to'),
    # "[IO]" or "[I0]" → "[10]"  (capital-I and capital-O misread as digits)
    (re.compile(r'\[IO\]'), '[10]'),
    (re.compile(r'\[I0\]'), '[10]'),
    # Capital-O misread as zero at end of a number: "209.00O" → "209.000"
    # Matches O preceded by a digit and followed by a word boundary.
    (re.compile(r'(?<=\d)O\b'), '0'),
    # Also catch O between digits: "2O16" → "2016"
    (re.compile(r'(?<=\d)O(?=\d)'), '0'),
    # Trailing escaped underscore "\\_" → "." (OCR reads sentence-end periods
    # as "_"; escape_latex_chars then turns "_" into "\\_").
    # r'\\\\_' matches the literal 2-char sequence backslash+underscore.
    (re.compile(r'\\\\_(?=\s|$)'), '.'),
    # Unescaped stray trailing underscore before space/end
    (re.compile(r'_(?=\s|$)'), '.'),
    # Superscript notation: "10-3" → "$10^{-3}$" (learning rate exponents)
    (re.compile(r'(?<!\$)\b(1\.5)\s*[xX]\s*10-(\d)\b'), r'$\1\\times10^{-\2}$'),
    (re.compile(r'(?<!\$)\b5\s*[xX]\s*10-(\d)\b'),      r'$5\\times10^{-\1}$'),
    (re.compile(r'(?<!\$)\b10-(\d)\b'),                   r'$10^{-\1}$'),
]


def _clean_ocr(text: str) -> str:
    """
    Post-processing of raw EasyOCR output to fix common OCR artifacts.

    1. Apply targeted character-confusion fixes (O/0, I/1, t/to, etc.)
    2. Rejoin soft-hyphenated line-breaks: "con - taining" → "containing"
    3. Fix thousands-separator dot confusion: "7.352" → "7,352"
    4. Strip leading/trailing whitespace.
    """
    if not text:
        return text
    for pattern, replacement in _OCR_FIXES:
        text = pattern.sub(replacement, text)
    text = _SOFT_HYPHEN_RE.sub(r'\1\2', text)
    text = _THOUSANDS_DOT_RE.sub(r'\1,\2', text)
    return text.strip()


# Fix 3 — Bullet list split: YOLO sometimes detects all bullet items as one
# List-item region. When the content of a single \item contains "Model A",
# "Model B" etc., split it into individual items.
_BULLET_SPLIT_RE = re.compile(
    r'(?<!\A)'                        # not at very start of string
    r'(?=\bModel\s+[A-D][\s:\-\.])',  # lookahead: "Model A:", "Model A-", "Model A."
)


def _split_bullet_items(text: str) -> List[str]:
    """
    Split a single OCR blob that contains multiple bullet points into
    individual item strings.  Returns a list with one entry per item;
    if no split point is found, returns a single-element list.

    Detects two patterns:
      • "Model A: … Model B: …"  (all items on one line from EasyOCR)
      • Items already separated by newlines
    """
    parts = _BULLET_SPLIT_RE.split(text)
    # Also try newline splitting as secondary strategy
    if len(parts) == 1 and '\n' in text:
        parts = [p.strip() for p in text.split('\n') if p.strip()]
    return [p.strip() for p in parts if p.strip()]


# ----------------------------------------------------------------
# YOLO class name → LaTeX wrapping function
# Each function receives: content (str) → returns: str (LaTeX)
#
# Fix C — Section-header uses \subsection*{} not \section{}
#   IEEE journal section labels ("D. TRAINING PROTOCOL",
#   "IV. IMPLEMENTATION DETAILS") are subsection-level headings,
#   not top-level \section commands.  \section creates large numbered
#   headings that override the journal's own numbering scheme and
#   promote OCR garbles into prominent document structure.
#
# Fix C — Title uses \begin{center}\textbf{\large ...}\end{center}
#   \title/\maketitle expects \author + \date; journal page extracts
#   don't have that context, so \maketitle would crash or produce a
#   malformed title block.
# ----------------------------------------------------------------

LATEX_WRAPPERS = {
    "Title": lambda c: (
        f"\n\\begin{{center}}\n"
        f"\\textbf{{\\large {_clean_ocr(c)}}}\n"
        f"\\end{{center}}\n"
    ),
    "Section-header": lambda c: f"\n\\subsection*{{{_clean_ocr(c)}}}\n",
    "Caption":        lambda c: f"\n\\textit{{{_clean_ocr(c)}}}\n",
    "Footnote":       lambda c: f"\\footnote{{{_clean_ocr(c)}}}",
    "Page-footer":    lambda c: f"% [footer: {c}]",
    "Page-header":    lambda c: f"% [header: {c}]",
    "Text":           lambda c: f"\n{_clean_ocr(c)}\n",
    # Fix E / Fix 3: each List-item is its own \item on a dedicated line.
    # _split_bullet_items handles the case where YOLO detects all bullets
    # as one region — the blob is split on "Model A/B/C/D" boundaries and
    # each sub-item gets its own \item prefix, joined by newlines.
    "List-item": lambda c: "\n".join(
        f"\\item {_clean_ocr(part)}"
        for part in _split_bullet_items(c)
    ),
    # Fix D: Formula wrapper — content is a LaTeX math string from
    # run_math_recognition, or an \includegraphics fallback if math
    # OCR returned empty.  The two cases need different environments.
    "Formula": lambda c: (
        f"\n\\begin{{equation}}\n{c}\n\\end{{equation}}\n"
        if c and not c.startswith("\\includegraphics")
        else f"\n\\begin{{center}}\n{c}\n\\end{{center}}\n"
    ),
    "Table": lambda c: (
        f"\n\\begin{{center}}\n"
        f"\\resizebox{{\\columnwidth}}{{!}}{{\n{c}\n}}\n"
        f"\\end{{center}}\n"
    ),
    "Picture": lambda c: (
        f"\n\\begin{{center}}\n"
        f"\\includegraphics[width=0.8\\linewidth]{{{c}}}\n"
        f"\\end{{center}}\n"
    ),
}


def wrap_content(class_name: str, content: str) -> str:
    """
    Wrap recognized content with correct LaTeX markup.
    Falls back to plain paragraph if class_name is unrecognized.
    """
    wrapper = LATEX_WRAPPERS.get(class_name)
    if wrapper:
        return wrapper(content)
    return f"\n{_clean_ocr(content)}\n"


def _build_body(parts: List[str], list_regions) -> str:
    """
    Build a LaTeX body string from parts, grouping consecutive
    list-items into itemize environments.

    Fix E: each \\item is already on its own line from wrap_content.
    We join with newlines so each bullet renders as a separate item,
    not a single concatenated blob.
    """
    assembled = []
    in_list = False
    for i, part in enumerate(parts):
        is_list_item = i in list_regions
        if is_list_item and not in_list:
            assembled.append("\\begin{itemize}")
            in_list = True
        elif not is_list_item and in_list:
            assembled.append("\\end{itemize}")
            in_list = False
        assembled.append(part)
    if in_list:
        assembled.append("\\end{itemize}")
    return "\n".join(assembled)


def assemble_document(
    body_parts: List[str],
    list_regions,
    is_two_column: bool = False,
    left_parts: List[str] = None,
    left_list_regions=None,
    right_parts: List[str] = None,
    right_list_regions=None,
    header_logo: Optional[str] = None,
) -> str:
    """
    Assemble full compilable .tex document from body parts.

    Parameters
    ----------
    body_parts         : full-width regions (above columns, or all content
                         for single-column pages)
    list_regions       : set/list of indices in body_parts that are List-items
    is_two_column      : use paracol for independent left/right columns
    left_parts         : left column content  (paracol mode only)
    left_list_regions  : indices in left_parts that are List-items
    right_parts        : right column content (paracol mode only)
    right_list_regions : indices in right_parts that are List-items
    header_logo        : filename of a logo to place right-aligned ABOVE
                         the column block (fix B).  None = no logo.

    Returns compilable .tex string ready for pdflatex.
    """
    preamble = (
        "\\documentclass{article}\n"
        "\\usepackage[margin=2cm]{geometry}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{ragged2e}\n"
        "\\setlength{\\emergencystretch}{3em}\n"
    )
    if is_two_column:
        preamble += "\\usepackage{paracol}\n"
    preamble += "\\begin{document}\n\\sloppy\n"
    closing = "\n\\end{document}\n"

    # Fix B: header logo block — right-aligned flush to the right margin,
    # placed BEFORE the paracol block so it appears at the top of the page
    # as it does in the original journal layout.  A thin \\hrule below it
    # replicates the separator line used by IEEE Access and similar journals.
    logo_block = ""
    if header_logo:
        logo_block = (
            "\n\\noindent\\hfill"
            f"\\includegraphics[height=1.8em]{{{header_logo}}}"
            "\\par\\noindent\\hrule\\vspace{4pt}\n"
        )

    if is_two_column and left_parts is not None:
        full_body  = _build_body(body_parts, list_regions)
        left_body  = _build_body(left_parts,  left_list_regions  or [])
        right_body = _build_body(right_parts  or [], right_list_regions or [])
        body = (
            logo_block
            + full_body + "\n"
            "\\begin{paracol}{2}\n"
            "\\RaggedRight\n"
            + left_body
            + "\n\\switchcolumn\n"
            "\\RaggedRight\n"
            + right_body
            + "\n\\end{paracol}\n"
        )
    else:
        body = logo_block + _build_body(body_parts, list_regions)

    return preamble + body + closing


def save_tex(content: str, output_path: str) -> None:
    """Write the assembled .tex to disk."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[✓] LaTeX file written: {os.path.abspath(output_path)}")