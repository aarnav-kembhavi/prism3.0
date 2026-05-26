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
  G — IEEE Citation bracket fixes added to clean up 121 -> [12] hallucinations
"""

import re
from typing import List, Optional
import os


# ----------------------------------------------------------------
# Post-processing: clean OCR output before wrapping in LaTeX
# ----------------------------------------------------------------

_SOFT_HYPHEN_RE = re.compile(r'(\w) ?- +([a-z])')
_THOUSANDS_DOT_RE = re.compile(r'\b(\d{1,3})\.(\d{3})\b')

_OCR_FIXES = [
    (re.compile(r'\bt0\b'), 'to'),
    (re.compile(r'\[IO\]'), '[10]'),
    (re.compile(r'\[I0\]'), '[10]'),
    (re.compile(r'(?<=\d)O\b'), '0'),
    (re.compile(r'(?<=\d)O(?=\d)'), '0'),
    
    # --- IEEE Citation & Bracket Hallucination Fixes ---
    (re.compile(r'\b1(\d{1,2})1\b'), r'[\1]'),               # 1121 -> [12]
    (re.compile(r'\bl(\d{1,2})l\b'), r'[\1]'),               # l13l -> [13]
    (re.compile(r'\[i(\d{1,2})\]', re.IGNORECASE), r'[\1]'), # [i6] -> [16]
    (re.compile(r'\[(\d{1,2})1\b'), r'[\1]'),                # [171 -> [17]
    (re.compile(r'\b1(\d{1,2})\]'), r'[\1]'),                # 115] -> [15]
    
    # --- IEEE Section Header Fixes ---
    # Fixes missing space after section letter (e.g., "DRETRIEVAL" -> "D. RETRIEVAL")
    (re.compile(r'^([A-Z])([A-Z]{3,})'), r'\1. \2'),
    
    # I (capital-I) misread as digit 1 in numeric contexts
    (re.compile(r'\bI(\d)'), r'1\1'),       
    (re.compile(r'(\d)I(\d)'), r'\g<1>1\2'), 
    (re.compile(r'(\d)I\b'), r'\g<1>1'),    
    # S misread as 8 in numeric/suffix contexts
    (re.compile(r'\bIS([A-Z])'), r'18\1'),   
    (re.compile(r'(\d)S([KMGkm%])'), r'\g<1>8\2'),  
    (re.compile(r'\\\\_(?=\s|$)'), '.'),
    (re.compile(r'_(?=\s|$)'), '.'),
    (re.compile(r'(?<!\$)\b(1\.5)\s*[xX]\s*10-(\d)\b'), r'$\1\\times10^{-\2}$'),
    (re.compile(r'(?<!\$)\b(5)\s*[xX]\s*10-(\d)\b'),      r'$5\\times10^{-\2}$'),
    (re.compile(r'(?<!\$)\b10-(\d)\b'),                   r'$10^{-\1}$'),
]


# IEEE section header splitter — splits merged detections like:
# "IV. RESULTS A. OVERALL COMPARISON" → two separate \subsection* entries
_IEEE_HEADER_SPLIT_RE = re.compile(
    r'(?<!\A)'                                           
    r'(?='                                               
    r'(?:(?<![IVXLCDMivxlcdm])[IVX]+\.[ \t])'          
    r'|(?<![A-Z])[A-Z]\.[ \t]'                          
    r')',
    re.VERBOSE,
)


def _split_section_headers(text: str) -> List[str]:
    parts = _IEEE_HEADER_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _clean_ocr(text: str) -> str:
    if not text:
        return text
    for pattern, replacement in _OCR_FIXES:
        text = pattern.sub(replacement, text)
    text = _SOFT_HYPHEN_RE.sub(r'\1\2', text)
    text = _THOUSANDS_DOT_RE.sub(r'\1,\2', text)
    return text.strip()


_BULLET_SPLIT_RE = re.compile(
    r'(?<!\A)'                        
    r'(?=\bModel\s+[A-D][\s:\-\.])',  
)


def _split_bullet_items(text: str) -> List[str]:
    parts = _BULLET_SPLIT_RE.split(text)
    if len(parts) == 1 and '\n' in text:
        parts = [p.strip() for p in text.split('\n') if p.strip()]
    return [p.strip() for p in parts if p.strip()]


LATEX_WRAPPERS = {
    "Title": lambda c: (
        f"\n\\begin{{center}}\n"
        f"\\textbf{{\\large {_clean_ocr(c)}}}\n"
        f"\\end{{center}}\n"
    ),
    "Section-header": lambda c: "\n".join(
        f"\n\\subsection*{{{_clean_ocr(part)}}}" for part in _split_section_headers(c)
    ) + "\n",
    "Caption":        lambda c: f"\n\\textit{{{_clean_ocr(c)}}}\n",
    "Footnote":       lambda c: f"\\footnote{{{_clean_ocr(c)}}}",
    "Page-footer":    lambda c: f"% [footer: {c}]",
    "Page-header":    lambda c: f"% [header: {c}]",
    "Text":           lambda c: f"\n{_clean_ocr(c)}\n",
    "List-item": lambda c: "\n".join(
        f"\\item {_clean_ocr(part)}"
        for part in _split_bullet_items(c)
    ),
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
    wrapper = LATEX_WRAPPERS.get(class_name)
    if wrapper:
        return wrapper(content)
    return f"\n{_clean_ocr(content)}\n"


def _build_body(parts: List[str], list_regions) -> str:
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
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[✓] LaTeX file written: {os.path.abspath(output_path)}")