"""
math_worker_unimernet.py
------------------------
Persistent subprocess for math OCR via UniMERNet-Small (dynamic INT8).

Drop-in replacement for MathOCRWorkerOnnx:
    from math_worker_unimernet import MathOCRWorkerUnimernet
    worker = MathOCRWorkerUnimernet()
    worker.start()
    results, counter = worker.run_math_batch(crops, figures_dir, counter)
    worker.stop()

The subprocess loads UniMERNet-Small (~810 MB FP32 -> 160 MB INT8) with
torch.ao.quantization.quantize_dynamic, keeping torch out of the main process.
"""

import os
import re
import sys
import multiprocessing as mp

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import numpy as np
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

_REPEAT_TILDE_RE  = re.compile(r'(~\s*){10,}')
_REPEAT_NEG_RE    = re.compile(r'(\\![\s\\!]*){10,}')
_REPEAT_QQUAD_RE  = re.compile(r'(\\q{0,1}quad\s*){5,}')
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
                text = text[:idx] + text[idx + 1:]
    bc = text.count(r'\begin{array}'); ec = text.count(r'\end{array}')
    if bc > ec:
        text = text.rstrip() + r'\end{array}' * (bc - ec)
    lc = text.count(r'\left'); rc = text.count(r'\right')
    if lc > rc:
        text = text.rstrip() + r'\right.' * (lc - rc)
    return text.strip()


def _worker_main(conn):
    """Loads UniMERNet-Small INT8 once, serves batches until shutdown."""
    import torch
    import unimernet
    from omegaconf import OmegaConf
    from unimernet.common.registry import registry
    from unimernet.processors import load_processor

    pkg_dir = os.path.dirname(unimernet.__file__)
    cfg = OmegaConf.load(os.path.join(pkg_dir, 'configs', 'models', 'unimernet_base.yaml'))

    model_dir = os.path.join(ROOT_DIR, 'unimernet_small_model')
    cfg.model.tokenizer_config.path = model_dir
    OmegaConf.update(cfg, 'model.model_config.model_name', 'wanderkid/unimernet_small')

    model = registry.get_model_class('unimernet').from_config(cfg.model)
    ckpt = torch.load(
        os.path.join(model_dir, 'unimernet_small.pth'),
        map_location='cpu',
        weights_only=False,
    )
    model.load_state_dict(ckpt['model'])
    model.eval()

    model = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    vis_processor = load_processor(
        'formula_image_eval',
        cfg=OmegaConf.create({'image_size': [192, 672]}),
    )

    conn.send('ready')

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg is None:
            break

        crop_arrays, figures_dir, math_counter_val = msg
        crops = [Image.fromarray(a) for a in crop_arrays]

        results: list[str] = []
        for crop in crops:
            try:
                pixel = vis_processor(crop)
                with torch.no_grad():
                    out = model.generate({'image': pixel.unsqueeze(0)})
                raw = out['pred_str'][0].strip() if out.get('pred_str') else ''
            except Exception as e:
                raw = ''
                print(f'[UniMERNet worker] inference error: {e}', file=sys.stderr)

            # strip outer math delimiters if present
            for delim in ('$$', '$', r'\[', r'\]', r'\(', r'\)'):
                if raw.startswith(delim):
                    raw = raw[len(delim):]
                if raw.endswith(delim):
                    raw = raw[:-len(delim)]
            clean = _sanitize(raw.strip())

            if clean:
                results.append(clean)
            elif figures_dir:
                math_counter_val += 1
                fname = f'formula_{math_counter_val:03d}.png'
                try:
                    crop.save(os.path.join(figures_dir, fname))
                    results.append(
                        f'\\includegraphics[width=0.5\\linewidth]{{{fname}}}'
                    )
                except Exception:
                    results.append('')
            else:
                results.append('')

        conn.send((results, math_counter_val))


class MathOCRWorkerUnimernet:
    """Persistent UniMERNet-Small INT8 subprocess — same interface as MathOCRWorkerOnnx."""

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
        print(f'[*] Math OCR worker (UniMERNet-Small INT8) started (PID {self._proc.pid})')

    def stop(self):
        if self._proc is None:
            return
        try:
            self._conn.send(None)
        except Exception:
            pass
        self._proc.join(timeout=30)
        if self._proc.is_alive():
            self._proc.terminate()
        self._proc = None
        self._conn = None
        print('[*] Math OCR worker (UniMERNet-Small INT8) stopped')

    def run_math_batch(
        self,
        crops: list,
        figures_dir: str,
        math_counter_val: int,
    ) -> tuple[list[str], int]:
        if not crops:
            return [], math_counter_val
        arrays = [np.array(c.convert('RGB')) for c in crops]
        self._conn.send((arrays, figures_dir, math_counter_val))
        return self._conn.recv()
