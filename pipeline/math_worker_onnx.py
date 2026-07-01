"""
math_worker_onnx.py
-------------------
Persistent subprocess for math OCR via Texo ONNX runtime.

Drop-in replacement for MathOCRWorker (math_worker.py) that uses ONNX
Runtime instead of PyTorch.  The main process never imports torch.

Benefits vs the torch worker:
  - Subprocess startup: ~0.4 s (vs ~3.3 s loading torch + FormulaNet)
  - Subprocess peak RSS: ~200 MB (vs ~500 MB with torch)
  - No INT8 quality regression (ONNX FP32 inference)

Interface is identical to MathOCRWorker:
    from math_worker_onnx import MathOCRWorkerOnnx
    worker = MathOCRWorkerOnnx()
    worker.start()
    results, counter = worker.run_math_batch(crops, figures_dir, counter)
    worker.stop()
"""

import os
import re
import sys
import multiprocessing as mp

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import numpy as np
from PIL import Image, ImageOps

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ensure child processes spawned by this module can import pipeline.*
_pythonpath = os.environ.get('PYTHONPATH', '')
if ROOT_DIR not in _pythonpath.split(os.pathsep):
    os.environ['PYTHONPATH'] = ROOT_DIR + (os.pathsep + _pythonpath if _pythonpath else '')

# ── sanitize helpers (no torch) ───────────────────────────────────────────────

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


# Variant-controlled thresholds (change these to run different variants)
_TILDE_THRESHOLD = 10   # Var-B: lower to 6 to catch more hallucinations
_COL_MIN_GAP     = 15   # Var-A: lower to 8 to capture tighter column gaps

# Cap on tokens generated per formula. Greedy decode is one sequential
# decoder.run() per token on CPU, and the quality gate runs only AFTER the
# full generation, so an over-long cap makes hallucinating crops (and their
# row/col-split retries) extremely expensive. 256 covers real display
# equations while bounding worst-case latency (was 384).
_MAX_NEW_TOKENS = 256

# Patterns that indicate Texo is looping / hallucinating
_BAD_PATTERNS = [
    re.compile(r'(\\hline\s*){5,}'),                          # repeated \hline
    re.compile(r'(\\cdot\s*){8,}'),                            # repeated \cdot
    re.compile(r'(\\quad\s*){8,}'),                            # repeated \quad
    re.compile(r'(&\s*\{\s*\}\s*){5,}'),                       # 5+ empty array cells
    re.compile(r'\\mathrm\s*\{[^}]*(~\s*){5,}'),              # spaced-char hallucination inside \mathrm
]


def _quality_gate(text: str, crop_w: int, crop_h: int) -> bool:
    """Return True if the LaTeX output looks valid, False if it should be discarded.

    Targets only the most obvious Texo failure modes:
    - Known repetition patterns (hline loops, nested arrays, etc.)
    - Extreme over-generation relative to crop size (>20× generous estimate)
    """
    if not text:
        return False

    # Known garbage patterns
    for pat in _BAD_PATTERNS:
        if pat.search(text):
            return False

    # Spaced-character hallucination: Texo renders text char-by-char with ~ separators.
    # Legitimate math formulas rarely exceed 8 tildes; 10+ is a near-certain hallucination.
    if text.count('~') >= _TILDE_THRESHOLD:
        return False

    # Extreme over-generation: only flag truly absurd cases
    # Expected max ≈ crop_area / 50 chars, with floor of 80
    crop_area = crop_w * crop_h
    expected_max = max(80, crop_area / 50)
    if len(text) > 10 * expected_max:
        return False

    return True


# ── row-splitting fallback ────────────────────────────────────────────────────

def _split_rows(img: Image.Image, min_content_h: int = 8, density_frac: float = 0.05) -> list:
    """Split a formula image at horizontal whitespace gaps between lines.

    Returns a list of sub-crop PIL Images. If no meaningful split is found,
    returns a list with just the original image.
    """
    arr = np.array(img.convert('L'))
    row_density = (arr < 200).sum(axis=1)
    thresh = max(1, row_density.max() * density_frac)
    is_gap = row_density < thresh

    regions, in_content, start = [], False, 0
    for r, gap in enumerate(is_gap):
        if not gap and not in_content:
            in_content, start = True, r
        elif gap and in_content:
            in_content = False
            if r - start >= min_content_h:
                regions.append((start, r))
    if in_content and len(arr) - start >= min_content_h:
        regions.append((start, len(arr)))

    if len(regions) <= 1:
        return [img]
    return [img.crop((0, s, img.width, e)) for s, e in regions]


def _split_cols(img: Image.Image, min_gap_w: int = 15, min_content_w: int = 10,
                density_frac: float = 0.05) -> list:
    """Split a formula image at large vertical whitespace gaps between sub-expressions.

    Finds contiguous content regions, merges those separated by gaps < min_gap_w,
    and splits at gaps >= min_gap_w. Returns a list with the original image when
    no meaningful split exists.
    """
    arr = np.array(img.convert('L'))
    col_density = (arr < 200).sum(axis=0)
    thresh = max(1, col_density.max() * density_frac)
    is_content = col_density >= thresh

    regions, in_c, start = [], False, 0
    for c, content in enumerate(is_content):
        if content and not in_c:
            in_c, start = True, c
        elif not content and in_c:
            in_c = False
            regions.append([start, c])
    if in_c:
        regions.append([start, len(is_content)])

    if not regions:
        return [img]

    # merge regions whose separating gap is narrower than min_gap_w
    merged = [regions[0]]
    for s, e in regions[1:]:
        if s - merged[-1][1] < min_gap_w:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    merged = [(s, e) for s, e in merged if e - s >= min_content_w]

    if len(merged) <= 1:
        return [img]
    return [img.crop((s, 0, e, img.height)) for s, e in merged]


# ── preprocessing ─────────────────────────────────────────────────────────────

_UNIMERNET_MEAN = 0.7931
_UNIMERNET_STD  = 0.1738
_IMG_SIZE       = 384


def _crop_margin(img: Image.Image) -> Image.Image:
    import cv2
    data = np.array(img.convert('L'), dtype=np.uint8)
    max_v, min_v = int(data.max()), int(data.min())
    if max_v == min_v:
        return img
    norm = (data.astype(np.float32) - min_v) / (max_v - min_v) * 255
    gray = (255 * (norm < 200)).astype(np.uint8)
    coords = cv2.findNonZero(gray)
    if coords is None:
        return img
    a, b, w, h = cv2.boundingRect(coords)
    return img.crop((a, b, w + a, h + b))


def _preprocess_to_tensor(crop: Image.Image) -> np.ndarray:
    """Returns float32 ndarray [1, 3, 384, 384] for encoder input.

    Replicates EvalMERImageProcessor without torch:
      Otsu binarize → crop_margin → resize (longer side = 384) → center-pad →
      grayscale → normalize → CHW float32
    """
    import cv2

    # Otsu binarize (same as original _preprocess_formula)
    arr = np.array(crop.convert('L'), dtype=np.uint8)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = Image.fromarray(binary).convert('RGB')

    # crop whitespace margins
    img = _crop_margin(img)

    # resize so longer side = 384 (equivalent to F.resize(shorter→384) + thumbnail)
    w, h = img.size
    if w > 0 and h > 0:
        scale = _IMG_SIZE / max(w, h)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # center-pad to 384×384
    dw = _IMG_SIZE - img.width
    dh = _IMG_SIZE - img.height
    img = ImageOps.expand(img, (dw // 2, dh // 2, dw - dw // 2, dh - dh // 2))

    # grayscale → float → normalize
    gray = np.array(img.convert('L'), dtype=np.float32) / 255.0
    gray = (gray - _UNIMERNET_MEAN) / _UNIMERNET_STD

    # [1, 3, H, W] float32 (repeat grayscale to 3 channels)
    chw = np.stack([gray, gray, gray], axis=0)          # [3, H, W]
    return chw[np.newaxis].astype(np.float32)            # [1, 3, H, W]


# ── ONNX autoregressive generate ─────────────────────────────────────────────

def _onnx_encode(enc_sess, pixels: np.ndarray) -> np.ndarray:
    """Run the vision encoder on a batch of pixels [N,3,384,384] → [N,seq,hid]."""
    (enc_hidden,) = enc_sess.run(['last_hidden_state'], {'pixel_values': pixels})
    return enc_hidden


def _onnx_generate(
    enc_sess,
    dec_sess,
    pixel: np.ndarray,
    tokenizer,
    max_new_tokens: int = 256,
    rep_penalty: float = 1.15,
) -> list[int]:
    """Encode one image then greedy-decode. Used for one-off (retry) crops."""
    enc_hidden = _onnx_encode(enc_sess, pixel)
    return _onnx_decode(dec_sess, enc_hidden, tokenizer, max_new_tokens, rep_penalty)


def _onnx_decode(
    dec_sess,
    enc_hidden: np.ndarray,
    tokenizer,
    max_new_tokens: int = 256,
    rep_penalty: float = 1.15,
) -> list[int]:
    """Autoregressive greedy decode from a single image's encoder hidden state.

    Uses use_cache_branch=False for step 0 (init pass) then True for subsequent
    steps, with one critical fix: the cross-attention (encoder) KV is frozen
    after step 0 and reused throughout.  Re-computing cross-attn KV at each
    cached step causes divergence; this matches Optimum's ORTModelForVision2Seq
    behaviour and gives numerically identical output to FP32 Texo.

    enc_hidden must be [1, seq, hid] (a single image's slice of a batch encode).
    """
    LAYERS   = 2
    HEADS    = 16
    HEAD_DIM = 24
    BOS = tokenizer.token_to_id('<s>')
    EOS = tokenizer.token_to_id('</s>')

    empty_dec = np.zeros((1, HEADS, 0, HEAD_DIM), dtype=np.float32)
    empty_enc = np.zeros((1, HEADS, 0, HEAD_DIM), dtype=np.float32)

    past_dec_k = [empty_dec] * LAYERS
    past_dec_v = [empty_dec] * LAYERS
    past_enc_k = [empty_enc] * LAYERS   # frozen after step 0
    past_enc_v = [empty_enc] * LAYERS   # frozen after step 0

    input_ids = np.array([[BOS]], dtype=np.int64)
    generated: list[int] = []

    for step in range(max_new_tokens):
        use_cache = step > 0
        feed = {
            'input_ids':             input_ids,
            'encoder_hidden_states': enc_hidden,
            'use_cache_branch':      np.array([use_cache]),
        }
        for l in range(LAYERS):
            feed[f'past_key_values.{l}.decoder.key']   = past_dec_k[l]
            feed[f'past_key_values.{l}.decoder.value'] = past_dec_v[l]
            feed[f'past_key_values.{l}.encoder.key']   = past_enc_k[l]
            feed[f'past_key_values.{l}.encoder.value'] = past_enc_v[l]

        out = dec_sess.run(None, feed)
        logits = out[0][0, -1].copy()   # logits at last position [vocab]

        # repetition penalty
        if rep_penalty != 1.0 and generated:
            for tok in set(generated):
                if logits[tok] < 0:
                    logits[tok] *= rep_penalty
                else:
                    logits[tok] /= rep_penalty

        next_token = int(np.argmax(logits))

        # Update decoder self-attention KV; freeze cross-attention after step 0
        past_dec_k = [out[1], out[5]]
        past_dec_v = [out[2], out[6]]
        if step == 0:
            past_enc_k = [out[3], out[7]]   # freeze: never update again
            past_enc_v = [out[4], out[8]]

        input_ids = np.array([[next_token]], dtype=np.int64)
        if next_token == EOS:
            break
        generated.append(next_token)

    return generated


def _onnx_decode_batch(
    dec_sess,
    enc_hidden: np.ndarray,
    tokenizer,
    max_new_tokens: int = 256,
    rep_penalty: float = 1.15,
) -> list[list[int]]:
    """Greedy-decode B formulas in parallel from a batched encoder output.

    enc_hidden: [B, seq, hid]. All B sequences step together; a sequence that
    emits EOS stops being recorded but keeps occupying its batch row until the
    whole batch finishes (or max_new_tokens). Because attention never crosses
    batch rows, each row's logits are bit-identical to a standalone decode, so
    the output matches per-crop greedy decoding exactly — while collapsing
    sum(lengths) sequential decoder calls into ~max(length) batched calls.
    """
    LAYERS   = 2
    HEADS    = 16
    HEAD_DIM = 24
    B = int(enc_hidden.shape[0])
    BOS = tokenizer.token_to_id('<s>')
    EOS = tokenizer.token_to_id('</s>')

    empty_dec = np.zeros((B, HEADS, 0, HEAD_DIM), dtype=np.float32)
    empty_enc = np.zeros((B, HEADS, 0, HEAD_DIM), dtype=np.float32)
    past_dec_k = [empty_dec] * LAYERS
    past_dec_v = [empty_dec] * LAYERS
    past_enc_k = [empty_enc] * LAYERS
    past_enc_v = [empty_enc] * LAYERS

    input_ids = np.full((B, 1), BOS, dtype=np.int64)
    generated: list[list[int]] = [[] for _ in range(B)]
    finished = np.zeros(B, dtype=bool)

    for step in range(max_new_tokens):
        use_cache = step > 0
        feed = {
            'input_ids':             input_ids,
            'encoder_hidden_states': enc_hidden,
            'use_cache_branch':      np.array([use_cache]),
        }
        for l in range(LAYERS):
            feed[f'past_key_values.{l}.decoder.key']   = past_dec_k[l]
            feed[f'past_key_values.{l}.decoder.value'] = past_dec_v[l]
            feed[f'past_key_values.{l}.encoder.key']   = past_enc_k[l]
            feed[f'past_key_values.{l}.encoder.value'] = past_enc_v[l]

        out = dec_sess.run(None, feed)
        logits = out[0][:, -1, :].copy()   # [B, vocab]

        if rep_penalty != 1.0:
            for b in range(B):
                if finished[b] or not generated[b]:
                    continue
                row = logits[b]
                for tok in set(generated[b]):
                    if row[tok] < 0:
                        row[tok] *= rep_penalty
                    else:
                        row[tok] /= rep_penalty

        next_tokens = logits.argmax(1).astype(np.int64)   # [B]

        past_dec_k = [out[1], out[5]]
        past_dec_v = [out[2], out[6]]
        if step == 0:
            past_enc_k = [out[3], out[7]]
            past_enc_v = [out[4], out[8]]

        for b in range(B):
            if finished[b]:
                continue
            if next_tokens[b] == EOS:
                finished[b] = True
            else:
                generated[b].append(int(next_tokens[b]))

        input_ids = next_tokens.reshape(B, 1)
        if finished.all():
            break

    return generated


# ── worker entry point (runs in subprocess) ───────────────────────────────────

def _worker_main(conn):
    """Loads Texo ONNX sessions once, serves batches until shutdown."""
    import onnxruntime as ort
    from tokenizers import Tokenizer  # rust-based, no torch

    # suppress onnxruntime verbose config warnings
    ort.set_default_logger_severity(3)

    MODEL_DIR = os.path.join(ROOT_DIR, 'Texo', 'model')
    ONNX_DIR  = os.path.join(MODEL_DIR, 'onnx')

    tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))

    from pipeline.onnx_config import apply_session_threads
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    apply_session_threads(opts)
    enc_sess = ort.InferenceSession(
        os.path.join(ONNX_DIR, 'encoder_model.onnx'),
        sess_options=opts,
        providers=['CPUExecutionProvider'],
    )
    dec_sess = ort.InferenceSession(
        os.path.join(ONNX_DIR, 'decoder_model_merged.onnx'),
        sess_options=opts,
        providers=['CPUExecutionProvider'],
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

        # Batched encode + batched greedy decode: one encoder pass and
        # ~max(length) decoder passes per chunk instead of per crop. All crops
        # in a chunk decode in parallel (attention never crosses batch rows, so
        # output is identical to per-crop greedy). Chunked to bound RAM and cap
        # padding waste from length variance. Retries below stay per-crop.
        pixels = [_preprocess_to_tensor(c) for c in crops]   # each [1,3,384,384]
        token_ids_all: list[list[int]] = []
        _BATCH = 8
        for _s in range(0, len(pixels), _BATCH):
            _batch = np.concatenate(pixels[_s:_s + _BATCH], axis=0)
            _eh = _onnx_encode(enc_sess, _batch)             # [b, seq, hid]
            token_ids_all.extend(_onnx_decode_batch(
                dec_sess, _eh, tokenizer,
                max_new_tokens=_MAX_NEW_TOKENS, rep_penalty=1.15))

        results: list[str] = []
        for i, crop in enumerate(crops):
            token_ids = token_ids_all[i]
            raw = tokenizer.decode(token_ids).strip()   # tokenizers skips specials by default

            # strip outer delimiters
            for delim in ('$$', '$', r'\[', r'\]', r'\(', r'\)'):
                if raw.startswith(delim): raw = raw[len(delim):]
                if raw.endswith(delim):   raw = raw[:-len(delim)]
            clean = _sanitize(raw)

            # Quality gate: discard over-generated or repetitive output
            if clean and not _quality_gate(clean, crop.width, crop.height):
                clean = ''

            def _run_texo(sub_crop):
                """Run Texo on sub_crop; return clean LaTeX or '' on gate failure."""
                px = _preprocess_to_tensor(sub_crop)
                ids = _onnx_generate(enc_sess, dec_sess, px, tokenizer,
                                     max_new_tokens=_MAX_NEW_TOKENS, rep_penalty=1.15)
                r = tokenizer.decode(ids).strip()
                for d in ('$$', '$', r'\[', r'\]', r'\(', r'\)'):
                    if r.startswith(d): r = r[len(d):]
                    if r.endswith(d):   r = r[:-len(d)]
                s = _sanitize(r.strip())
                return s if (s and _quality_gate(s, sub_crop.width, sub_crop.height)) else ''

            def _col_results(sub_crop):
                """Run col-split (using _COL_MIN_GAP) and return joined pieces, or ''."""
                cols = _split_cols(sub_crop, min_gap_w=_COL_MIN_GAP)
                if len(cols) <= 1:
                    return ''
                pieces = [p for c in cols if (p := _run_texo(c))]
                return ' '.join(pieces) if pieces else ''

            # Row-splitting fallback: if full image failed, try splitting into lines.
            # Each row that itself fails the quality gate also gets a col-split attempt.
            if not clean:
                rows = _split_rows(crop)
                if len(rows) > 1:
                    row_results = []
                    for row in rows:
                        row_clean = _run_texo(row)
                        if not row_clean:
                            row_clean = _col_results(row)
                        if row_clean:
                            row_results.append(row_clean)
                    if row_results:
                        clean = r' \\ '.join(row_results)

            # Column-splitting fallback: if row-split also failed, try splitting wide
            # formulas into horizontal sub-expressions at large whitespace gaps
            if not clean:
                clean = _col_results(crop)

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


# ── MathOCRWorkerOnnx class ───────────────────────────────────────────────────

class MathOCRWorkerOnnx:
    """Persistent Texo ONNX subprocess — same interface as MathOCRWorker."""

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
        self._conn.recv()   # wait for 'ready'
        print(f'[*] Math OCR worker (ONNX) started (PID {self._proc.pid})')

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
        print('[*] Math OCR worker (ONNX) stopped')

    def run_math_batch(
        self,
        crops: list,
        figures_dir: str,
        math_counter_val: int,
    ) -> tuple[list[str], int]:
        """Returns (latex_strings, updated_counter). Same as MathOCRWorker."""
        if not crops:
            return [], math_counter_val
        arrays = [np.array(c.convert('RGB')) for c in crops]
        self._conn.send((arrays, figures_dir, math_counter_val))
        return self._conn.recv()


class MathOCRWorkerOnnxDual:
    """Two MathOCRWorkerOnnx subprocesses that split crops in parallel.

    Halves math OCR latency on formula-heavy pages.
    API-compatible with MathOCRWorkerOnnx.
    """

    def __init__(self):
        self._w1 = MathOCRWorkerOnnx()
        self._w2 = MathOCRWorkerOnnx()

    def start(self):
        self._w1.start()
        self._w2.start()

    def stop(self):
        self._w1.stop()
        self._w2.stop()

    def run_math_batch(
        self,
        crops: list,
        figures_dir: str,
        math_counter_val: int,
    ) -> tuple[list[str], int]:
        if not crops:
            return [], math_counter_val
        if len(crops) == 1:
            return self._w1.run_math_batch(crops, figures_dir, math_counter_val)
        from concurrent.futures import ThreadPoolExecutor
        mid = len(crops) // 2
        # Give w2 a counter offset of mid to prevent formula image filename collisions
        with ThreadPoolExecutor(max_workers=2) as exe:
            f1 = exe.submit(self._w1.run_math_batch, crops[:mid], figures_dir, math_counter_val)
            f2 = exe.submit(self._w2.run_math_batch, crops[mid:], figures_dir, math_counter_val + mid)
            r1, _c1 = f1.result()
            r2, _c2 = f2.result()
        return r1 + r2, math_counter_val + len(crops)
