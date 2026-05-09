"""
latex_builder.py
----------------
Maps YOLO class names to LaTeX wrappers and assembles the document.
This is the ONLY file that knows about LaTeX syntax.
"""

import os
import re

def wrap_content(class_name: str, content: str) -> str:
    """Wraps raw string content into LaTeX environment based on YOLO class."""
    if not content or content.strip() == "":
        return ""
    
    # Clean OCR artefacts (common to all text classes)
    content = _clean_ocr(content)

    if class_name == "Title":
        return f"\\begin{{center}}\n\\huge\\bfseries {content}\n\\end{{center}}\n\n"
    
    elif class_name == "Section-header":
        # DocLayNet Section-header is usually a subsection in a paper
        return f"\\subsection*{{{content}}}\n\n"
    
    elif class_name == "Text":
        return f"{content}\n\n"
    
    elif class_name == "Caption":
        return f"\\begin{{center}}\n\\small\\textit{{{content}}}\n\\end{{center}}\n\n"
    
    elif class_name == "Footnote":
        return f"\\vfill\\hrule\\vspace{{2pt}}\n\\noindent\\scriptsize {content}\n"
    
    elif class_name == "Formula":
        # Texo returns plain LaTeX without delimiters
        return f"\\begin{{equation*}}\n{content}\n\\end{{equation*}}\n\n"
    
    elif class_name == "List-item":
        # Grouped by assemble_document, here we just return the item text
        return content

    elif class_name == "Picture":
        # 'content' is the filename
        return f"\\begin{{center}}\n\\includegraphics[width=0.8\\linewidth]{{{content}}}\n\\end{{center}}\n\n"

    elif class_name == "Table":
        # 'content' is already a full \begin{tabular}... environment
        return f"\\begin{{center}}\n{content}\n\\end{{center}}\n\n"

    return f"{content}\n\n"


def assemble_document(body_parts: list[str], list_regions: set[int], 
                      is_two_column: bool = False,
                      left_parts: list[str] = None,
                      left_list_regions: set[int] = None,
                      right_parts: list[str] = None,
                      right_list_regions: set[int] = None,
                      header_logo: str = None) -> str:
    """Assembles the final compilable LaTeX source."""
    preamble = [
        "\\documentclass{article}",
        "\\usepackage[margin=2cm]{geometry}",
        "\\usepackage{amsmath}",
        "\\usepackage{graphicx}",
        "\\usepackage{booktabs}",
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage{ragged2e}",
        "\\setlength{\\emergencystretch}{3em}",
    ]
    
    if is_two_column:
        preamble.append("\\usepackage{paracol}")

    doc_start = ["\\begin{document}", "\\sloppy"]
    
    # Optional Header Logo (Right Aligned above content)
    if header_logo:
        doc_start.append(f"\n\\noindent\\hfill\\includegraphics[height=1.8em]{{{header_logo}}}\\par\\noindent\\hrule\\vspace{{4pt}}\n")

    def build_body(parts, list_indices):
        if not parts: return ""
        lines = []
        in_list = False
        for i, p in enumerate(parts):
            if i in list_indices:
                if not in_list:
                    lines.append("\\begin{itemize}")
                    in_list = True
                lines.append(f"  \\item {p}")
            else:
                if in_list:
                    lines.append("\\end{itemize}")
                    in_list = False
                lines.append(p)
        if in_list:
            lines.append("\\end{itemize}")
        return "\n".join(lines)

    content = []
    if is_two_column:
        content.append("\\begin{paracol}{2}")
        content.append("\\RaggedRight")
        content.append(build_body(left_parts, left_list_regions or set()))
        content.append("\\switchcolumn")
        content.append("\\RaggedRight")
        content.append(build_body(right_parts, right_list_regions or set()))
        content.append("\\end{paracol}")
    else:
        content.append(build_body(body_parts, list_regions))

    doc_end = ["\\end{document}"]
    
    return "\n".join(preamble + doc_start + content + doc_end)


def _clean_ocr(text: str) -> str:
    """Fix common OCR artifacts and LaTeX special characters."""
    if not text: return ""
    
    # 1. Join soft hyphens (end of line)
    text = re.sub(r'(\w)-\s*\n(\w)', r'\1\2', text)
    
    # 2. Fix thousands separator confusion (1.000 -> 1000 in numbers)
    text = re.sub(r'(\d)\.(\d{3})', r'\1\2', text)
    
    # 3. Fix O vs 0 confusion in numbers
    text = re.sub(r'(\d)O', r'\1 0', text)
    text = re.sub(r'O(\d)', r'0\1', text)

    # 4. Standard LaTeX Escapes
    # (Note: models_interface already does some escaping, but we double-check)
    for char, rep in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#")]:
        if char in text and f"\\{char}" not in text:
            text = text.replace(char, rep)

    return text

def save_tex(content: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[✓] LaTeX file written: {os.path.abspath(path)}")
