"""
text_worker.py
--------------
Persistent subprocess for text + table OCR via RapidOCR.

By running RapidOCR in a separate process (no torch imports), the worker's
memory footprint stays at ~130-160 MB instead of inheriting the ~400 MB
torch/CUDA baseline of the main process.

Usage from the main process:
    from text_worker import TextOCRWorker
    worker = TextOCRWorker()
    worker.start()
    texts  = worker.run_text_batch(crops, is_screenshot=False)
    tables = worker.run_table_batch(crops)
    worker.stop()

This module must be top-level (not nested in __main__) so 'spawn' can
pickle the worker function on Windows.
"""

import io
import os
import sys
import statistics
import multiprocessing as mp

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import numpy as np
from PIL import Image, ImageOps, ImageFilter

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── helpers replicated from models_interface (no torch dependency) ────────────

def _escape_latex(text: str) -> str:
    if not text:
        return text
    for ch, rep in [('&', r'\&'), ('%', r'\%'), ('$', r'\$'), ('#', r'\#'),
                    ('_', r'\_'), ('{', r'\{'), ('}', r'\}'),
                    ('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}')]:
        if ch in text:
            text = text.replace(ch, rep)
    return text


def _filter_nonascii(text: str) -> str:
    if all(ord(c) < 128 for c in text):
        return text
    filtered = ''.join(c for c in text if ord(c) < 128)
    return ' '.join(filtered.split())


def _reconstruct_lines(res) -> str:
    SPACE_GAP_FACTOR = 0.35
    LINE_MERGE_FRAC  = 0.6
    if not res:
        return ""
    items = []
    for entry in res:
        bbox, text = entry[0], entry[1]
        try:
            xs = [pt[0] for pt in bbox]; ys = [pt[1] for pt in bbox]
        except (TypeError, IndexError):
            xs = [bbox[0], bbox[2]]; ys = [bbox[1], bbox[3]]
        x_left = min(xs); x_right = max(xs)
        y_top = min(ys); y_bot = max(ys)
        height = max(y_bot - y_top, 1)
        items.append({'text': text, 'x_left': x_left, 'x_right': x_right,
                      'y_ctr': (y_top + y_bot) / 2.0, 'height': height})
    if not items:
        return ""
    items.sort(key=lambda d: d['y_ctr'])
    lines, current_line = [], [items[0]]
    for item in items[1:]:
        threshold = min(current_line[-1]['height'], item['height']) * LINE_MERGE_FRAC
        if abs(item['y_ctr'] - current_line[-1]['y_ctr']) <= threshold:
            current_line.append(item)
        else:
            lines.append(current_line); current_line = [item]
    lines.append(current_line)
    out = []
    for line in lines:
        line.sort(key=lambda d: d['x_left'])
        char_widths = [(d['x_right'] - d['x_left']) / max(len(d['text']), 1) for d in line]
        median_cw = statistics.median(char_widths) if char_widths else 8.0
        space_thresh = SPACE_GAP_FACTOR * median_cw
        parts = [line[0]['text']]
        for j in range(1, len(line)):
            if line[j]['x_left'] - line[j-1]['x_right'] >= space_thresh:
                parts.append(' ')
            parts.append(line[j]['text'])
        out.append(''.join(parts))
    return '\n'.join(out)


def _stitch_and_run(engine, processed_nps):
    SEP = 20
    max_w = max(p.shape[1] for p in processed_nps)
    total_h = sum(p.shape[0] for p in processed_nps) + SEP * (len(processed_nps) - 1)
    stitched = np.full((total_h, max_w, 3), 255, dtype=np.uint8)
    y_ranges, y = [], 0
    for p in processed_nps:
        h, w = p.shape[:2]
        stitched[y:y + h, :w] = p
        y_ranges.append((y, y + h))
        y += h + SEP
    stitched = stitched[:y - SEP]
    res, _ = engine(stitched)
    per_crop = [[] for _ in processed_nps]
    if res:
        for entry in res:
            bbox_poly, text, conf = entry[0], entry[1], entry[2]
            try:
                cy = sum(pt[1] for pt in bbox_poly) / len(bbox_poly)
            except (TypeError, IndexError):
                cy = (bbox_poly[1] + bbox_poly[3]) / 2
            for ci, (y_start, y_end) in enumerate(y_ranges):
                if y_start <= cy < y_end:
                    adj = [[pt[0], pt[1] - y_start] for pt in bbox_poly]
                    per_crop[ci].append([adj, text, conf])
                    break
    return per_crop


def _sauvola_binarize(gray_np: np.ndarray, window: int = 31, k: float = 0.2, R: float = 128.0) -> np.ndarray:
    """
    Sauvola local adaptive binarization.

    Threshold per pixel: T = mean * (1 + k * (std/R - 1))
    Pixels above T are background (white); below T are foreground (black text).
    Better than Otsu/global thresholds for colored or textured backgrounds
    (newsprint, magazine pages, section sidebars) because the threshold
    adapts to local brightness variations.
    """
    from scipy.ndimage import uniform_filter
    img = gray_np.astype(np.float64)
    mean = uniform_filter(img, window)
    mean_sq = uniform_filter(img ** 2, window)
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0.0))
    threshold = mean * (1.0 + k * (std / R - 1.0))
    return (img > threshold).astype(np.uint8) * 255


def _preprocess_crop(crop: Image.Image, is_screenshot: bool) -> np.ndarray:
    border   = 10 if is_screenshot else 30
    max_side = 960 if is_screenshot else 1500

    if is_screenshot:
        result = crop.convert('RGB')
        # Apply Sauvola when the background is non-white (magazine/newspaper
        # colored section backgrounds, gradients, photo fill areas).
        # 90th-percentile of grayscale approximates the background level;
        # values well below 255 indicate a colored or tinted background.
        grey_np = np.array(crop.convert('L'), dtype=np.uint8)
        background_level = float(np.percentile(grey_np, 90))
        if background_level < 225:
            binary_np = _sauvola_binarize(grey_np)
            result = Image.fromarray(binary_np).convert('RGB')
    else:
        import cv2
        grey = crop.convert('L')
        arr  = np.array(grey, dtype=np.float32)
        rms_contrast = float(arr.std())
        result = crop.convert('RGB')
        if rms_contrast < 40:
            grey_eq = ImageOps.autocontrast(grey, cutoff=1)
            result  = Image.merge('RGB', [grey_eq, grey_eq, grey_eq])
            grey    = grey_eq
        result = result.filter(ImageFilter.UnsharpMask(radius=1.5, percent=180, threshold=3))
        grey_sharp = result.convert('L')
        grey_np    = np.array(grey_sharp, dtype=np.uint8)
        rms_after  = float(grey_np.astype(np.float32).std())
        background_level = float(np.percentile(grey_np, 90))
        if rms_after < 40:
            binary_np = cv2.adaptiveThreshold(
                grey_np, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=25, C=12,
            )
            result = Image.fromarray(binary_np).convert('RGB')
        elif background_level < 210:
            # Normal contrast but non-white background (newsprint, colored band) —
            # Sauvola adapts to local brightness, recovering text from tinted paper.
            binary_np = _sauvola_binarize(grey_np)
            result = Image.fromarray(binary_np).convert('RGB')

    padded  = ImageOps.expand(result, border=border, fill='white')
    max_dim = max(padded.width, padded.height)
    if max_dim > max_side:
        scale  = max_side / max_dim
        padded = padded.resize(
            (int(padded.width * scale), int(padded.height * scale)),
            Image.Resampling.LANCZOS,
        )
    return np.array(padded)


def _table_heuristic(tokens, img_w):
    """Identical table heuristic from models_interface."""
    if not tokens:
        return ""
    ALIGN_TOL = 0.03
    rows = {}
    for tok in tokens:
        y_key = round(tok['y_ctr'] / max(img_w * 0.015, 1))
        rows.setdefault(y_key, []).append(tok)
    lines = []
    for y_key in sorted(rows):
        row = sorted(rows[y_key], key=lambda t: t['x_left'])
        lines.append(' & '.join(t['text'] for t in row))
    return ' \\\\\n'.join(lines)


def _tokens_from_result(result, img_w):
    if not result:
        return []
    tokens = []
    for entry in result:
        bbox, text, conf = entry[0], entry[1], entry[2]
        try:
            xs = [pt[0] for pt in bbox]; ys = [pt[1] for pt in bbox]
        except (TypeError, IndexError):
            xs = [bbox[0], bbox[2]]; ys = [bbox[1], bbox[3]]
        tokens.append({'text': text, 'x_left': min(xs), 'x_right': max(xs),
                       'y_ctr': (min(ys) + max(ys)) / 2.0})
    return tokens


# ── worker main loop ──────────────────────────────────────────────────────────

def _worker_main(conn):
    """Entry point for the OCR worker subprocess. Runs until shutdown signal."""
    # Re-apply arena + memory-mapping patch in subprocess (fresh Python state)
    try:
        import onnxruntime as _ort
        _orig = _ort.InferenceSession; _opts = _ort.SessionOptions
        def _patched(p, sess_options=None, providers=None, **kw):
            if sess_options is None: sess_options = _opts()
            sess_options.enable_cpu_mem_arena = False
            try:
                sess_options.add_session_config_entry("session.use_memory_mapped_if_possible", "1")
            except Exception:
                pass
            return _orig(p, sess_options=sess_options, providers=providers, **kw)
        _ort.InferenceSession = _patched
    except Exception:
        pass

    from rapidocr_onnxruntime import RapidOCR
    base_kwargs = dict(det_limit_type='max', det_limit_side_len=1280)
    en_rec  = os.path.join(ROOT_DIR, 'en_PP-OCRv4_rec.onnx')
    en_dict = os.path.join(ROOT_DIR, 'en_dict.txt')
    if os.path.exists(en_rec) and os.path.exists(en_dict):
        base_kwargs['rec_model_path'] = en_rec
        base_kwargs['rec_keys_path']  = en_dict
    # Screenshots: higher threshold + no CLS (text always horizontal in screenshots)
    engine_screenshot = RapidOCR(**base_kwargs, det_box_thresh=0.5, use_cls=False)
    engine_photo      = RapidOCR(**base_kwargs, det_box_thresh=0.3)
    # CJK engine: bundled ch_PP-OCRv4 models (char dict embedded in model)
    cjk_kwargs = dict(det_limit_type='max', det_limit_side_len=1280)
    engine_cjk_screenshot = RapidOCR(**cjk_kwargs, det_box_thresh=0.5, use_cls=False)
    engine_cjk_photo      = RapidOCR(**cjk_kwargs, det_box_thresh=0.3)
    conn.send('ready')

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg is None:
            break

        task, payload = msg

        if task == 'text':
            crop_arrays, is_screenshot = payload
            crops = [Image.fromarray(a) for a in crop_arrays]
            processed = [_preprocess_crop(c, is_screenshot) for c in crops]
            engine = engine_screenshot if is_screenshot else engine_photo
            results = [''] * len(crops)
            chunk_size = 20
            for start in range(0, len(crops), chunk_size):
                chunk = processed[start:start + chunk_size]
                per_crop = _stitch_and_run(engine, chunk)
                for ci, matches in enumerate(per_crop):
                    if matches:
                        txt = _reconstruct_lines(matches)
                        results[start + ci] = _escape_latex(_filter_nonascii(txt))
            conn.send(results)

        elif task == 'text_cjk':
            crop_arrays, is_screenshot = payload
            crops = [Image.fromarray(a) for a in crop_arrays]
            processed = [_preprocess_crop(c, is_screenshot) for c in crops]
            engine = engine_cjk_screenshot if is_screenshot else engine_cjk_photo
            results = [''] * len(crops)
            chunk_size = 20
            for start in range(0, len(crops), chunk_size):
                chunk = processed[start:start + chunk_size]
                per_crop = _stitch_and_run(engine, chunk)
                for ci, matches in enumerate(per_crop):
                    if matches:
                        # CJK text flows continuously — strip intra-block newlines
                        txt = _reconstruct_lines(matches).replace('\n', '')
                        results[start + ci] = _escape_latex(txt)
            conn.send(results)

        elif task == 'text_mixed':
            # English-first with per-block CJK fallback for mixed-language pages.
            # Runs English engine on all blocks; any block returning < 3 chars
            # is retried with the CJK engine (likely Chinese text).
            crop_arrays, is_screenshot = payload
            crops = [Image.fromarray(a) for a in crop_arrays]
            processed = [_preprocess_crop(c, is_screenshot) for c in crops]
            en_engine  = engine_screenshot if is_screenshot else engine_photo
            cjk_engine = engine_cjk_screenshot if is_screenshot else engine_cjk_photo
            results = [''] * len(crops)
            chunk_size = 20
            # English pass
            for start in range(0, len(crops), chunk_size):
                chunk = processed[start:start + chunk_size]
                per_crop = _stitch_and_run(en_engine, chunk)
                for ci, matches in enumerate(per_crop):
                    if matches:
                        txt = _reconstruct_lines(matches)
                        results[start + ci] = _escape_latex(_filter_nonascii(txt))
            # CJK fallback for blocks where English returned < 3 visible chars
            cjk_indices = [i for i, r in enumerate(results) if len(r.strip()) < 3]
            for start in range(0, len(cjk_indices), chunk_size):
                batch_idx = cjk_indices[start:start + chunk_size]
                chunk = [processed[i] for i in batch_idx]
                per_crop = _stitch_and_run(cjk_engine, chunk)
                for ci, matches in enumerate(per_crop):
                    if matches:
                        txt = _reconstruct_lines(matches).replace('\n', '')
                        results[batch_idx[ci]] = _escape_latex(txt)
            conn.send(results)

        elif task == 'probe':
            # Run English OCR on a small sample; return total output char count.
            # If near-zero, caller should switch to CJK mode.
            crop_arrays, is_screenshot = payload
            crops = [Image.fromarray(a) for a in crop_arrays]
            processed = [_preprocess_crop(c, is_screenshot) for c in crops]
            engine = engine_screenshot if is_screenshot else engine_photo
            total_chars = 0
            if processed:
                per_crop = _stitch_and_run(engine, processed)
                for matches in per_crop:
                    if matches:
                        total_chars += len(_reconstruct_lines(matches))
            conn.send(total_chars)

        elif task == 'table':
            crop_arrays = payload
            crops = [Image.fromarray(a) for a in crop_arrays]
            nps   = [np.array(c.convert('RGB')) for c in crops]
            per_crop = _stitch_and_run(engine, nps)
            results = []
            for np_img, crop_result in zip(nps, per_crop):
                tokens = _tokens_from_result(crop_result, np_img.shape[1])
                results.append(_table_heuristic(tokens, np_img.shape[1]) if tokens else '')
            conn.send(results)


# ── TextOCRWorker class (used in main process) ────────────────────────────────

class TextOCRWorker:
    """Manages a single persistent RapidOCR worker subprocess."""

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
        ack = parent_conn.recv()  # 'ready'
        self._conn = parent_conn
        print(f'[*] Text OCR worker started (PID {self._proc.pid})')

    def stop(self):
        if self._proc is None:
            return
        try:
            self._conn.send(None)
        except Exception:
            pass
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.terminate()
        self._proc = None
        self._conn = None
        print('[*] Text OCR worker stopped')

    @staticmethod
    def _serialize(crops):
        return [np.array(c.convert('RGB')) for c in crops]

    def run_text_batch(self, crops, is_screenshot=False):
        if not crops:
            return []
        self._conn.send(('text', (self._serialize(crops), is_screenshot)))
        return self._conn.recv()

    def run_text_batch_cjk(self, crops, is_screenshot=False):
        if not crops:
            return []
        self._conn.send(('text_cjk', (self._serialize(crops), is_screenshot)))
        return self._conn.recv()

    def run_text_batch_mixed(self, crops, is_screenshot=False):
        """English-first with per-block CJK fallback (for en_ch_mixed pages)."""
        if not crops:
            return []
        self._conn.send(('text_mixed', (self._serialize(crops), is_screenshot)))
        return self._conn.recv()

    def run_language_probe(self, crops, is_screenshot=False):
        """Run English OCR on a sample crop; returns total char count.
        Near-zero → likely CJK page."""
        if not crops:
            return 0
        sample = crops[:3]
        self._conn.send(('probe', (self._serialize(sample), is_screenshot)))
        return self._conn.recv()

    def run_table_batch(self, crops):
        if not crops:
            return []
        self._conn.send(('table', self._serialize(crops)))
        return self._conn.recv()


class TextOCRWorkerDual:
    """Two TextOCRWorker subprocesses that split crops in parallel.

    Halves recognition latency on large batches by sending the first half to
    worker A and the second half to worker B, then merging results in order.
    API-compatible with TextOCRWorker.
    """

    def __init__(self):
        self._w1 = TextOCRWorker()
        self._w2 = TextOCRWorker()

    def start(self):
        self._w1.start()
        self._w2.start()

    def stop(self):
        self._w1.stop()
        self._w2.stop()

    def run_text_batch(self, crops, is_screenshot=False):
        if not crops:
            return []
        if len(crops) == 1:
            return self._w1.run_text_batch(crops, is_screenshot)
        from concurrent.futures import ThreadPoolExecutor
        mid = len(crops) // 2
        with ThreadPoolExecutor(max_workers=2) as exe:
            f1 = exe.submit(self._w1.run_text_batch, crops[:mid], is_screenshot)
            f2 = exe.submit(self._w2.run_text_batch, crops[mid:], is_screenshot)
            return f1.result() + f2.result()

    def run_table_batch(self, crops):
        if not crops:
            return []
        if len(crops) == 1:
            return self._w1.run_table_batch(crops)
        from concurrent.futures import ThreadPoolExecutor
        mid = len(crops) // 2
        with ThreadPoolExecutor(max_workers=2) as exe:
            f1 = exe.submit(self._w1.run_table_batch, crops[:mid])
            f2 = exe.submit(self._w2.run_table_batch, crops[mid:])
            return f1.result() + f2.result()
