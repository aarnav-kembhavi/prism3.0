# evaluation/normalizer.py
import re

# ── Rule 0a: PRISM layout artifact removal ────────────────────────────────────
# These constructs appear in PRISM output but never in GT.

_LOGO_LINE_RE     = re.compile(
    r'\\noindent\\hfill\\includegraphics\[[^\]]*\]\{[^}]*\}'
    r'\\par\\noindent\\hrule\\vspace\{[^}]*\}\s*'
)
_PARACOL_BEGIN_RE = re.compile(r'\\begin\{paracol\}\{[^}]*\}')
_PARACOL_END_RE   = re.compile(r'\\end\{paracol\}')
_SWITCHCOL_RE     = re.compile(r'\\switchcolumn\b')
_LAYOUT_CMD_RE    = re.compile(r'\\(?:RaggedRight|raggedright|sloppy|noindent|hfill|hrule)\b')
_VHSPACE_RE       = re.compile(r'\\[vh]space\*?\{[^}]*\}')
_PAR_RE           = re.compile(r'\\par\b')
_CENTER_RE        = re.compile(r'\\begin\{center\}|\\end\{center\}')
_ARRAY_RE         = re.compile(r'\\begin\{array\}\{[^}]*\}|\\end\{array\}')
_INCLUDEGFX_RE    = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}')
_COMMENT_RE       = re.compile(r'%[^\n]*')
_SETLENGTH_RE     = re.compile(r'\\setlength\{[^}]*\}\{[^}]*\}')
_LABEL_RE         = re.compile(r'\\label\{[^}]*\}')

# ── Rule 0b: Math semantic-neutral command removal ────────────────────────────
# These appear in both PRISM and GT but carry no mathematical content —
# they are spacing, style, or size hints that create spurious token differences.

# Math line-break separator (often left over after array stripping)
_MATH_LINEBREAK_RE   = re.compile(r'\\\\')
# Math spacing commands — zero semantic value
_MATH_SPACING_RE     = re.compile(r'\\(?:qquad|quad|enspace|thinspace|[,;:!])\b')
# Math style selectors — do not affect the expression itself
_MATH_STYLE_RE       = re.compile(r'\\(?:displaystyle|textstyle|scriptstyle|scriptscriptstyle)\b')
# Bracket-size prefixes — keep the bracket character, drop the size hint
_LEFT_RIGHT_RE       = re.compile(r'\\(?:left|right)\s*')
_BIGBRACKET_RE       = re.compile(r'\\(?:[Bb]igg?m?|bigg?m?)\s*')
# Equation modifiers — not part of the mathematical expression
_NONUMBER_RE         = re.compile(r'\\nonumber\b')
_HLINE_RE            = re.compile(r'\\hline\b')
# Prime normalisation: \prime inside braces → standalone apostrophe
_PRIME_BRACE_RE      = re.compile(r'\^\{\\prime\}')         # ^{\prime} → '
_PRIME_BARE_RE       = re.compile(r'\^\\prime\b')           # ^\prime   → '
# Drop braces around single alphanumeric chars in sub/superscripts
_SINGLE_SUB_RE       = re.compile(r'_\{([A-Za-z0-9])\}')   # _{x} → _x
_SINGLE_SUP_RE       = re.compile(r'\^\{([A-Za-z0-9])\}')  # ^{x} → ^x

# ── Rule 2: Math environment patterns ────────────────────────────────────────
_MATH_ENV_PATTERNS = [
    re.compile(r'\\begin\{equation\*?\}'),  re.compile(r'\\end\{equation\*?\}'),
    re.compile(r'\\begin\{eqnarray\*?\}'),  re.compile(r'\\end\{eqnarray\*?\}'),
    re.compile(r'\\begin\{gather\*?\}'),    re.compile(r'\\end\{gather\*?\}'),
    re.compile(r'\\begin\{align\*?\}'),     re.compile(r'\\end\{align\*?\}'),
    re.compile(r'\\\['), re.compile(r'\\\]'),
    re.compile(r'\\\('), re.compile(r'\\\)'),
]

# ── Rule 3: Section command patterns (including starred variants) ─────────────
_SECTION_PATTERNS = [
    re.compile(r'\\section\*?\{[^}]*\}'),
    re.compile(r'\\subsection\*?\{[^}]*\}'),
    re.compile(r'\\subsubsection\*?\{[^}]*\}'),
    re.compile(r'\\paragraph\*?\{[^}]*\}'),
    re.compile(r'\\chapter\*?\{[^}]*\}'),
]

# ── Rule 6: Math polymorphism ─────────────────────────────────────────────────
_SUBSUP_RE = re.compile(r'([a-zA-Z\\]+)\^\{([^}]*)\}_\{([^}]*)\}')

# ── Rule 0c: Math command synonym normalisation ───────────────────────────────
# Texo and GT use different-but-equivalent LaTeX commands. Normalise to one
# canonical form so edit distance is not penalised for style choices.

# Bold math: \boldsymbol{x} → \mathbf{x}  (same visual output for Latin/Greek)
_BOLDSYMBOL_RE    = re.compile(r'\\boldsymbol\b')
# Slanted inequality variants → standard forms
_LEQSLANT_RE      = re.compile(r'\\leqslant\b')
_GEQSLANT_RE      = re.compile(r'\\geqslant\b')
_NLEQSLANT_RE     = re.compile(r'\\nleqslant\b')
_NGEQSLANT_RE     = re.compile(r'\\ngeqslant\b')
# \operatorname* → \operatorname (asterisk = limits in display, no content diff)
_OPERATORNAME_RE  = re.compile(r'\\operatorname\*')
# Orphaned equation numbers left after \begin{array} stripping, e.g. {(1)} or (3)
_EQ_NUMBER_RE     = re.compile(r'\(\d+\)')
# \mathds → \mathbb (both used for blackboard-bold, e.g. identity matrix)
_MATHDS_RE        = re.compile(r'\\mathds\b')
# \iint, \iiint → \int\int, \int\int\int (same integral content)
_IINT_RE          = re.compile(r'\\iint\b')
_IIINT_RE         = re.compile(r'\\iiint\b')


def normalize_latex(latex_str, remove_spaces=True):
    """
    Normalization rules based on PDF2LaTeX (Wang & Liu, 2020), extended with:
      Rule 0a — strip PRISM layout artifacts not present in ground truth.
      Rule 0b — strip math semantic-neutral commands (spacing, style, size hints)
                that create spurious token differences without affecting content.
    """
    # ── Rule 0a: PRISM layout artifacts ──────────────────────────────────────
    latex_str = _LOGO_LINE_RE.sub('', latex_str)
    latex_str = _PARACOL_BEGIN_RE.sub('', latex_str)
    latex_str = _PARACOL_END_RE.sub('', latex_str)
    latex_str = _SWITCHCOL_RE.sub('', latex_str)
    latex_str = _LAYOUT_CMD_RE.sub('', latex_str)
    latex_str = _VHSPACE_RE.sub('', latex_str)
    latex_str = _PAR_RE.sub('', latex_str)
    latex_str = _CENTER_RE.sub('', latex_str)
    latex_str = _ARRAY_RE.sub('', latex_str)
    latex_str = _INCLUDEGFX_RE.sub('', latex_str)
    latex_str = _COMMENT_RE.sub('', latex_str)
    latex_str = _SETLENGTH_RE.sub('', latex_str)
    latex_str = _LABEL_RE.sub('', latex_str)

    # ── Rule 0b: Math semantic-neutral commands ───────────────────────────────
    # Order: prime normalisation before single-char brace dropping
    latex_str = _PRIME_BRACE_RE.sub("'", latex_str)
    latex_str = _PRIME_BARE_RE.sub("'", latex_str)
    latex_str = _SINGLE_SUB_RE.sub(r'_\1', latex_str)
    latex_str = _SINGLE_SUP_RE.sub(r'^\1', latex_str)
    latex_str = _MATH_LINEBREAK_RE.sub(' ', latex_str)
    latex_str = _MATH_SPACING_RE.sub('', latex_str)
    latex_str = _MATH_STYLE_RE.sub('', latex_str)
    latex_str = _LEFT_RIGHT_RE.sub('', latex_str)
    latex_str = _BIGBRACKET_RE.sub('', latex_str)
    latex_str = _NONUMBER_RE.sub('', latex_str)
    latex_str = _HLINE_RE.sub('', latex_str)

    # ── Rule 0c: Math command synonyms ───────────────────────────────────────
    latex_str = _BOLDSYMBOL_RE.sub(r'\\mathbf', latex_str)
    latex_str = _LEQSLANT_RE.sub(r'\\leq', latex_str)
    latex_str = _GEQSLANT_RE.sub(r'\\geq', latex_str)
    latex_str = _NLEQSLANT_RE.sub(r'\\nleq', latex_str)
    latex_str = _NGEQSLANT_RE.sub(r'\\ngeq', latex_str)
    latex_str = _OPERATORNAME_RE.sub(r'\\operatorname', latex_str)
    latex_str = _EQ_NUMBER_RE.sub('', latex_str)
    latex_str = _MATHDS_RE.sub(r'\\mathbb', latex_str)
    latex_str = _IIINT_RE.sub(r'\\int\\int\\int', latex_str)
    latex_str = _IINT_RE.sub(r'\\int\\int', latex_str)

    # ── Rule 1: Remove preamble (content before \begin{document}) ────────────
    if r'\begin{document}' in latex_str:
        latex_str = latex_str.split(r'\begin{document}', 1)[1]
    if r'\end{document}' in latex_str:
        latex_str = latex_str.split(r'\end{document}', 1)[0]

    # ── Rule 2: Replace all math environment claimers with $ ─────────────────
    latex_str = latex_str.replace(r'$$', '$')
    for pat in _MATH_ENV_PATTERNS:
        latex_str = pat.sub('$', latex_str)

    # ── Rule 3: Remove section/paragraph claimers (incl. starred variants) ───
    for pat in _SECTION_PATTERNS:
        latex_str = pat.sub('', latex_str)

    # ── Rule 4: Whitespace handling ───────────────────────────────────────────
    if remove_spaces:
        latex_str = re.sub(r'\s+', '', latex_str)
    else:
        latex_str = re.sub(r'\s+', ' ', latex_str).strip()

    # ── Rule 5: Lowercase everything ─────────────────────────────────────────
    latex_str = latex_str.lower()

    # ── Rule 6: Normalize math polymorphism (sub before super) ───────────────
    latex_str = normalize_math_polymorphism(latex_str)

    return latex_str


def normalize_math_polymorphism(latex_str):
    """Standardize X^{sup}_{sub} → X_{sub}^{sup} so both orderings compare equal."""
    def sort_sub_super(match):
        return f"{match.group(1)}_{{{match.group(3)}}}^{{{match.group(2)}}}"
    return _SUBSUP_RE.sub(sort_sub_super, latex_str)


def split_math_and_text(latex_str):
    """
    Split normalized LaTeX into math and plaintext parts.
    Math parts are delimited by $ ... $ (non-greedy, escaped-dollar-safe).
    """
    temp  = latex_str.replace(r'\$', '__ESCAPED_DOLLAR__')
    parts = re.split(r'(\$[^$]+\$)', temp)

    math_parts, text_parts = [], []
    for p in parts:
        if p.startswith('$') and p.endswith('$') and len(p) > 2:
            math_parts.append(p[1:-1].replace('__ESCAPED_DOLLAR__', r'\$'))
        else:
            text_parts.append(p.replace('__ESCAPED_DOLLAR__', r'\$'))

    return ''.join(math_parts), ''.join(text_parts)
