# evaluation/normalizer.py
import re

def normalize_latex(latex_str):
    """
    Apply the 6 normalization rules from PDF2LaTeX (Wang & Liu, 2020).
    """
    # Rule 1: Remove preamble (content before \begin{document})
    if r'\begin{document}' in latex_str:
        latex_str = latex_str.split(r'\begin{document}', 1)[1]
    if r'\end{document}' in latex_str:
        latex_str = latex_str.split(r'\end{document}', 1)[0]

    # Rule 2: Replace all math environment claimers with $
    math_envs = [
        r'\\begin\{equation\*?\}', r'\\end\{equation\*?\}',
        r'\\begin\{eqnarray\*?\}', r'\\end\{eqnarray\*?\}',
        r'\\begin\{gather\*?\}',   r'\\end\{gather\*?\}',
        r'\\begin\{align\*?\}',    r'\\end\{align\*?\}',
        r'\\\[', r'\\\]',
    ]
    for pattern in math_envs:
        latex_str = re.sub(pattern, '$', latex_str)

    # Rule 3: Remove section/paragraph claimers
    section_cmds = [
        r'\\section\{[^}]*\}', r'\\subsection\{[^}]*\}',
        r'\\subsubsection\{[^}]*\}', r'\\paragraph\{[^}]*\}',
        r'\\chapter\{[^}]*\}',
    ]
    for pattern in section_cmds:
        latex_str = re.sub(pattern, '', latex_str)

    # Rule 4: Remove all spaces
    latex_str = re.sub(r'\s+', '', latex_str)

    # Rule 5: Lowercase everything
    latex_str = latex_str.lower()

    # Rule 6: Normalize math polymorphism
    # e.g., X_a^b and X^b_a are equivalent
    latex_str = normalize_math_polymorphism(latex_str)

    return latex_str


def normalize_math_polymorphism(latex_str):
    """
    Handle cases where the same expression can be written multiple ways.
    Sorts sub/superscript order: always write _ before ^.
    """
    # Pattern: X^{...}_{...} → X_{...}^{...}  (standardize sub before super)
    def sort_sub_super(match):
        base = match.group(1)
        sup = match.group(2)
        sub = match.group(3)
        return f"{base}_{{{sub}}}^{{{sup}}}"

    # Match: base^{sup}_{sub}
    pattern = r'([a-zA-Z\\]+)\^\{([^}]*)\}_\{([^}]*)\}'
    latex_str = re.sub(pattern, sort_sub_super, latex_str)

    return latex_str


def split_math_and_text(latex_str):
    """
    Split normalized LaTeX into math and plaintext parts.
    Math parts are wrapped in $ ... $.
    """
    parts = re.split(r'(\$[^$]*\$)', latex_str)
    math_parts = ''.join(p[1:-1] for p in parts if p.startswith('$') and p.endswith('$'))
    text_parts = ''.join(p for p in parts if not (p.startswith('$') and p.endswith('$')))
    return math_parts, text_parts
