"""
pix2tex_worker.py
-----------------
Persistent subprocess for math OCR via pix2tex (LaTeX-OCR).

Drop-in replacement for MathOCRWorkerOnnx that uses pix2tex instead of Texo.
Requires: pip install pix2tex timm==0.5.4

Interface is identical to MathOCRWorkerOnnx:
    from pipeline.pix2tex_worker import Pix2TexWorker
    worker = Pix2TexWorker()
    worker.start()
    results, counter = worker.run_math_batch(crops, figures_dir, counter)
    worker.stop()
"""

import os
import sys
import multiprocessing as mp

import numpy as np
from PIL import Image

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_pythonpath = os.environ.get('PYTHONPATH', '')
if ROOT_DIR not in _pythonpath.split(os.pathsep):
    os.environ['PYTHONPATH'] = ROOT_DIR + (os.pathsep + _pythonpath if _pythonpath else '')


def _worker_main(conn):
    """Loads pix2tex LatexOCR once, serves batches until shutdown."""
    import warnings
    warnings.filterwarnings('ignore')
    os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

    from pix2tex.cli import LatexOCR
    model = LatexOCR()
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
                raw = model(crop)
                clean = raw.strip() if raw else ''
                # Strip outer delimiters pix2tex sometimes adds
                for delim in ('$$', '$', r'\[', r'\]', r'\(', r'\)'):
                    if clean.startswith(delim):
                        clean = clean[len(delim):]
                    if clean.endswith(delim):
                        clean = clean[:-len(delim)]
                clean = clean.strip()
            except Exception:
                clean = ''

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


class Pix2TexWorker:
    """Persistent pix2tex subprocess — same interface as MathOCRWorkerOnnx."""

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
        print(f'[*] Pix2Tex worker started (PID {self._proc.pid})')

    def stop(self):
        if self._proc is None:
            return
        try:
            self._conn.send(None)
        except Exception:
            pass
        self._proc.join(timeout=15)
        if self._proc.is_alive():
            self._proc.terminate()
        self._proc = None
        self._conn = None
        print('[*] Pix2Tex worker stopped')

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
