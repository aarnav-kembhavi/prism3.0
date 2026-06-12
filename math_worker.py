"""
math_worker.py
--------------
Persistent subprocess for math OCR via Texo (FormulaNet).

Keeps torch + FormulaNet isolated in a separate process so the main pipeline
process never imports PyTorch — saving ~350 MB from the main process RSS.

Also applies torch.ao.quantization.quantize_dynamic (INT8 Linear layers) to
FormulaNet before inference, halving weight memory inside the subprocess.

Usage from the main process:
    from math_worker import MathOCRWorker
    worker = MathOCRWorker()
    worker.start()
    results = worker.run_math_batch(crops, figures_dir, math_counter_val)
    # results: (list[str], updated_math_counter_val)
    worker.stop()
"""

import io
import os
import sys
import re
import multiprocessing as mp
import numpy as np
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, 'Texo', 'src'))

# ── sanitize / regex (no torch) ───────────────────────────────────────────────

_REPEAT_TILDE_RE = re.compile(r'(~\s*){10,}')
_REPEAT_NEG_RE   = re.compile(r'(\\![\s\\!]*){10,}')
_REPEAT_QQUAD_RE = re.compile(r'(\\q{0,1}quad\s*){5,}')
_ARROW_SUBS = [
    (re.compile(r'\\[Rr]arr\b'), r'\\rightarrow'),
    (re.compile(r'\\[Ll]arr\b'), r'\\leftarrow'),
    (re.compile(r'\\[Uu]arr\b'), r'\\uparrow'),
    (re.compile(r'\\[Dd]arr\b'), r'\\downarrow'),
]


def _sanitize(text: str) -> str:
    if not text:
        return text
    for pat, rep in _ARROW_SUBS:
        text = pat.sub(rep, text)
    text = _REPEAT_TILDE_RE.sub('~ ', text)
    text = _REPEAT_NEG_RE.sub(r'\\! ', text)
    text = _REPEAT_QQUAD_RE.sub(r'\\qquad ', text)
    open_c, close_c = text.count('{'), text.count('}')
    if open_c > close_c:
        text += '}' * (open_c - close_c)
    elif close_c > open_c:
        for _ in range(close_c - open_c):
            idx = text.rfind('}')
            if idx != -1:
                text = text[:idx] + text[idx+1:]
    bc = text.count(r'\begin{array}'); ec = text.count(r'\end{array}')
    if bc > ec:
        text = text.rstrip() + r'\end{array}' * (bc - ec)
    lc = text.count(r'\left'); rc = text.count(r'\right')
    if lc > rc:
        text = text.rstrip() + r'\right.' * (lc - rc)
    return text.strip()


def _preprocess_formula(crop: Image.Image) -> Image.Image:
    import cv2
    arr = np.array(crop.convert("L"), dtype=np.uint8)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary).convert("RGB")


# ── worker entry point (runs in subprocess) ───────────────────────────────────

def _worker_main(conn):
    """Loads Texo once, processes math batches on demand until shutdown."""
    # Windows DLL fix before torch import
    if sys.platform == 'win32':
        try:
            import torch as _t
            lib = os.path.join(os.path.dirname(_t.__file__), 'lib')
            if os.path.exists(lib) and hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(lib)
        except Exception:
            pass

    import torch
    from texo.data.processor import EvalMERImageProcessor
    from texo.model.formulanet import FormulaNet
    import texo.utils.config
    from transformers import AutoTokenizer

    MODEL_PATH = os.path.join(ROOT_DIR, 'Texo', 'model')
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_PATH)
    model      = FormulaNet.from_pretrained(MODEL_PATH)

    # INT8 dynamic quantization: halves Linear weight memory with no accuracy loss
    model = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    model.eval()

    processor = EvalMERImageProcessor(image_size={'width': 384, 'height': 384})
    conn.send('ready')

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg is None:
            break

        crop_bytes_list, figures_dir, math_counter_val = msg
        crops = [Image.open(io.BytesIO(b)) for b in crop_bytes_list]

        preprocessed  = [_preprocess_formula(c) for c in crops]
        pixel_tensors = torch.cat(
            [processor(p).unsqueeze(0) for p in preprocessed], dim=0
        )

        with torch.no_grad():
            outputs = model.generate(
                pixel_values       = pixel_tensors,
                max_new_tokens     = 150,
                repetition_penalty = 1.15,
            )

        results_raw = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results = []
        for i, raw in enumerate(results_raw):
            clean = raw.strip()
            for delim in ("$$", "$", r"\[", r"\]", r"\(", r"\)"):
                if clean.startswith(delim): clean = clean[len(delim):]
                if clean.endswith(delim):   clean = clean[:-len(delim)]
            clean = _sanitize(clean)
            if clean:
                results.append(clean)
            elif figures_dir:
                math_counter_val += 1
                fname = f"formula_{math_counter_val:03d}.png"
                try:
                    crops[i].save(os.path.join(figures_dir, fname))
                    results.append(
                        f"\\includegraphics[width=0.5\\linewidth]{{{fname}}}"
                    )
                except Exception:
                    results.append("")
            else:
                results.append("")

        conn.send((results, math_counter_val))


# ── MathOCRWorker class (used in main process) ────────────────────────────────

class MathOCRWorker:
    """Manages the persistent Texo math OCR subprocess."""

    def __init__(self):
        self._proc = None
        self._conn = None

    def start(self):
        if self._proc is not None:
            return
        ctx = mp.get_context('spawn')
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        self._proc = ctx.Process(target=_worker_main, args=(child_conn,), daemon=True)
        self._proc.start()
        child_conn.close()
        self._conn = parent_conn
        self._conn.recv()  # wait for 'ready'
        print(f'[*] Math OCR worker started (PID {self._proc.pid})')

    def stop(self):
        if self._proc is None:
            return
        try:
            self._conn.send(None)
        except Exception:
            pass
        self._proc.join(timeout=10)
        if self._proc.is_alive():
            self._proc.terminate()
        self._proc = None
        self._conn = None
        print('[*] Math OCR worker stopped')

    def run_math_batch(
        self,
        crops: list,
        figures_dir: str,
        math_counter_val: int,
    ) -> tuple[list[str], int]:
        """Returns (latex_strings, updated_counter)."""
        if not crops:
            return [], math_counter_val
        crop_bytes = []
        for c in crops:
            buf = io.BytesIO()
            c.save(buf, format='PNG')
            crop_bytes.append(buf.getvalue())
        self._conn.send((crop_bytes, figures_dir, math_counter_val))
        return self._conn.recv()
