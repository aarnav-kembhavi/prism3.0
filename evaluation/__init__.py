"""
evaluation — LaTeX output evaluation framework
Edit distance metrics, PDF2LaTeX normalization, math/text split scoring.
"""

from .eval import evaluate_page, evaluate_dataset
from .normalizer import normalize_latex, split_math_and_text
