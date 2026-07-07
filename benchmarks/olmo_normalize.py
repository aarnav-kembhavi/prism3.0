"""
olmo_normalize.py
-----------------
olmOCR-Bench emission normalizer for PRISM markdown.

olmOCR-Bench's math test is *presence-based*: a GT equation passes if some
delimited equation in the prediction renders (KaTeX) to the same visual form.
PRISM's LaTeX is tuned for OmniDocBench's CDM/pdflatex path and emits constructs
that are legal in pdflatex but (a) fail in KaTeX or (b) render to a different
visual form than the bare equation:

  * single/multi-row `\\begin{array}{..}{ EQ }\\end{array}` wrappers around what
    are really standalone display equations (KaTeX renders the array box, not the
    bare equation -> visual mismatch);
  * non-KaTeX macros (\\textcircled, orphan \\big., \\dph, empty groups).

This module rewrites each math span into its KaTeX-canonical form *in place*
(arrays unwrapped into their constituent equations, unsupported macros mapped).
It is applied only to the olmOCR-Bench candidate markdown; the OmniDocBench
pipeline and its emitted markdown are untouched.
"""
import re

_ARR_HDR = re.compile(r"\\begin\{array\}\s*\{[^}]*\}")
_ARR_END = re.compile(r"\\end\{array\}")
_DISP = re.compile(r"\\\[(.+?)\\\]", re.S)
_INLINE = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$", re.S)

# non-KaTeX / harmful macros -> KaTeX-safe replacement
_KATEX_FIX = [
    (re.compile(r"\\textcircled\s*\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\\textcircled"), ""),
    (re.compile(r"\\[bB]ig\."), ""),            # orphan \big. / \Big.
    (re.compile(r"\\dph\b"), ""),               # Texo garble token
    (re.compile(r"\\qquad|\\quad"), r" "),
    (re.compile(r"\\,|\\;|\\!|\\:"), r" "),      # thin spaces -> space
    (re.compile(r"\{\s*\}"), ""),                # empty groups
]


def _strip_outer_braces(s: str) -> str:
    s = s.strip()
    while len(s) >= 2 and s[0] == "{" and s[-1] == "}":
        depth = 0
        ok = True
        for i, c in enumerate(s):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    ok = False
                    break
        if ok:
            s = s[1:-1].strip()
        else:
            break
    return s


def _sanitize(eq: str) -> str:
    for pat, rep in _KATEX_FIX:
        eq = pat.sub(rep, eq)
    return re.sub(r"\s+", " ", eq).strip()


def unwrap_array(eq: str):
    """Return the list of constituent equations if eq is array-wrapped, else []."""
    if "\\begin{array}" not in eq:
        return []
    body = _ARR_END.sub("", _ARR_HDR.sub("", eq)).strip()
    body = _strip_outer_braces(body)
    rows = [r for r in re.split(r"\\\\", body) if r.strip()]
    out = []
    for r in rows:
        r = _strip_outer_braces(r).replace("&", " ")
        r = re.sub(r"\s+", " ", r).strip()
        if r:
            out.append(r)
    return out


def _canonical_forms(eq: str):
    """Best KaTeX-renderable representation(s) of a single equation."""
    forms = []
    rows = unwrap_array(eq)
    if rows:
        forms.extend(_sanitize(r) for r in rows)
    else:
        forms.append(_sanitize(eq))
    # de-dup, keep order
    seen = set(); out = []
    for f in forms:
        if f and f not in seen:
            seen.add(f); out.append(f)
    return out


def normalize_md(md: str) -> str:
    """Rewrite display + inline math spans into KaTeX-canonical form in place.
    A multi-row array display span expands into one \\[..\\] per row."""
    def _disp_sub(m):
        forms = _canonical_forms(m.group(1))
        if not forms:
            return m.group(0)
        return "\n".join(f"\\[ {f} \\]" for f in forms)

    def _inl_sub(m):
        forms = _canonical_forms(m.group(1))
        if not forms:
            return m.group(0)
        # inline stays inline; if an array produced multiple rows, join display
        if len(forms) == 1:
            return f"${forms[0]}$"
        return " ".join(f"${f}$" for f in forms)

    md = _DISP.sub(_disp_sub, md)
    md = _INLINE.sub(_inl_sub, md)
    return md


if __name__ == "__main__":
    import sys
    print(normalize_md(open(sys.argv[1], encoding="utf-8").read()))
