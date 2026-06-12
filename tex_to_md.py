"""
tex_to_md.py
-----------
Convert PRISM's LaTeX output to OmniDocBench-compatible Markdown.

Format expected by OmniDocBench:
  - Plain text paragraphs for text blocks
  - \\[...\\] blocks for display formulas
  - $...$ for inline math (PRISM doesn't produce inline math, so no-op)
  - HTML or LaTeX tables (we keep the tabular env as-is)
  - Section headings as # / ##
"""

import re


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_body(tex: str) -> str:
    m = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', tex, re.DOTALL)
    return m.group(1).strip() if m else tex.strip()


def _remove_logo_block(text: str) -> str:
    # \noindent\hfill\includegraphics[height=1.8em]{...}\par\noindent\hrule\vspace{4pt}
    text = re.sub(
        r'\\noindent\\hfill\\includegraphics\[[^\]]*\]\{[^}]*\}\\par\\noindent\\hrule\\vspace\{[^}]*\}\s*',
        '', text,
    )
    return text


def _flatten_paracol(text: str) -> str:
    text = re.sub(r'\\begin\{paracol\}\{[^}]*\}\s*', '', text)
    text = re.sub(r'\\end\{paracol\}\s*', '', text)
    text = re.sub(r'\\switchcolumn\s*', '\n', text)
    text = re.sub(r'\\RaggedRight\s*', '', text)
    return text


def _extract_group(text: str, start: int) -> tuple[str, int]:
    """Extract the content of a LaTeX group starting at '{' at position start."""
    if start >= len(text) or text[start] != '{':
        return '', start
    depth = 0
    i = start
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    return text[start + 1:], len(text)


def _strip_cmd(text: str, cmd: str, replacement_fn=None) -> str:
    """Replace \\cmd{...} with replacement_fn(inner) or just inner."""
    pattern = re.compile(re.escape(cmd) + r'\s*\{')
    result = []
    pos = 0
    for m in pattern.finditer(text):
        result.append(text[pos:m.start()])
        inner, end = _extract_group(text, m.end() - 1)
        result.append(replacement_fn(inner) if replacement_fn else inner)
        pos = end
    result.append(text[pos:])
    return ''.join(result)


# ── Environment converters ─────────────────────────────────────────────────────

def _convert_equations(text: str) -> str:
    """\\begin{equation}...\\end{equation} → \\[...\\]"""
    def repl(m):
        inner = m.group(1).strip()
        return f'\n\\[\n{inner}\n\\]\n'
    text = re.sub(
        r'\\begin\{equation\}(.*?)\\end\{equation\}',
        repl, text, flags=re.DOTALL,
    )
    # align, gather, etc. — also treat as display math
    for env in ('align', 'align*', 'gather', 'gather*', 'multline', 'multline*', 'flalign', 'flalign*'):
        text = re.sub(
            rf'\\begin\{{{re.escape(env)}\}}(.*?)\\end\{{{re.escape(env)}\}}',
            lambda m: f'\n\\[\n{m.group(1).strip()}\n\\]\n',
            text, flags=re.DOTALL,
        )
    return text


def _convert_array_blocks(text: str) -> str:
    """Remove \\begin{array}...\\end{array} if they're now bare (outside equation)."""
    return text


def _convert_center(text: str) -> str:
    """\\begin{center}...\\end{center} → content (stripped)."""
    def repl(m):
        inner = m.group(1).strip()
        # If it's \\resizebox wrapping a tabular, keep the tabular
        inner = re.sub(r'\\resizebox\{[^}]*\}\{[^}]*\}\{(.*?)\}', r'\1', inner, flags=re.DOTALL)
        return '\n' + inner + '\n'
    return re.sub(r'\\begin\{center\}(.*?)\\end\{center\}', repl, text, flags=re.DOTALL)


def _convert_itemize(text: str) -> str:
    """\\begin{itemize}...\\end{itemize} → markdown list."""
    def repl(m):
        inner = m.group(1)
        items = re.split(r'\\item\s*', inner)
        lines = []
        for it in items:
            it = it.strip()
            if it:
                lines.append('- ' + it.rstrip())
        return '\n' + '\n'.join(lines) + '\n'
    return re.sub(r'\\begin\{itemize\}(.*?)\\end\{itemize\}', repl, text, flags=re.DOTALL)


def _convert_sections(text: str) -> str:
    """\\subsection*{...} → ## ...\n and title centers → # ..."""
    # Title: \begin{center}\n\textbf{\large ...}\end{center}
    text = re.sub(
        r'\\begin\{center\}\s*\\textbf\{\\large\s*(.*?)\}\s*\\end\{center\}',
        lambda m: '\n# ' + m.group(1).strip() + '\n',
        text, flags=re.DOTALL,
    )
    # \section*{...}
    text = re.sub(r'\\section\*\{([^}]+)\}', lambda m: '\n# ' + m.group(1) + '\n', text)
    # \subsection*{...}
    text = re.sub(r'\\subsection\*\{([^}]+)\}', lambda m: '\n## ' + m.group(1) + '\n', text)
    return text


# ── Inline formatting ──────────────────────────────────────────────────────────

def _strip_formatting(text: str) -> str:
    """Remove LaTeX formatting commands, keep content."""
    text = _strip_cmd(text, r'\textbf', lambda s: f'**{s}**')
    text = _strip_cmd(text, r'\textit', lambda s: f'*{s}*')
    text = _strip_cmd(text, r'\emph', lambda s: f'*{s}*')
    text = _strip_cmd(text, r'\textit', lambda s: f'*{s}*')
    text = _strip_cmd(text, r'\large', lambda s: s)
    text = _strip_cmd(text, r'\footnote', lambda s: '')
    text = _strip_cmd(text, r'\textcolor', lambda s: s)  # imperfect but ok
    text = re.sub(r'\\includegraphics\[[^\]]*\]\{[^}]*\}', '', text)
    text = re.sub(r'\\label\{[^}]*\}', '', text)
    text = re.sub(r'\\ref\{[^}]*\}', '', text)
    text = re.sub(r'\\cite\{[^}]*\}', '', text)
    text = re.sub(r'\\vspace\{[^}]*\}', '', text)
    text = re.sub(r'\\hspace\{[^}]*\}', '', text)
    text = re.sub(r'\\noindent\b', '', text)
    text = re.sub(r'\\sloppy\b', '', text)
    text = re.sub(r'\\par\b', '\n\n', text)
    text = re.sub(r'\\linebreak\b', '\n', text)
    text = re.sub(r'\\newline\b', '\n', text)
    text = re.sub(r'\\\\(?!\[)', '\n', text)  # \\ newline (not \\[
    text = re.sub(r'\\hrule\b', '', text)
    text = re.sub(r'\\hfill\b', '', text)
    text = re.sub(r'\\centering\b', '', text)
    # Comments
    text = re.sub(r'%.*', '', text)
    return text


# ── Post-clean ────────────────────────────────────────────────────────────────

def _clean_whitespace(text: str) -> str:
    # Normalise many blank lines to max 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip lines that are only whitespace
    lines = [ln.rstrip() for ln in text.split('\n')]
    return '\n'.join(lines).strip()


# ── Main entrypoint ───────────────────────────────────────────────────────────

def tex_to_omnidocbench_md(tex_content: str) -> str:
    """Convert PRISM LaTeX output to OmniDocBench Markdown."""
    text = _extract_body(tex_content)
    text = _remove_logo_block(text)
    text = _flatten_paracol(text)
    text = _convert_equations(text)
    text = _convert_sections(text)
    text = _convert_itemize(text)
    text = _convert_center(text)
    text = _strip_formatting(text)
    text = _clean_whitespace(text)
    return text


if __name__ == '__main__':
    import sys
    content = open(sys.argv[1], encoding='utf-8').read()
    print(tex_to_omnidocbench_md(content))
