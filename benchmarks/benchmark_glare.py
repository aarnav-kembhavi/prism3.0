"""
benchmark_glare.py
------------------
Test PRISM on clean vs synthetically glared document images.
Measures per-page latency, peak RSS (main + all worker subprocesses),
character extraction rate, and NED vs OmniDocBench GT markdown.

Conditions:
  clean    — original 18 demo images unchanged
  specular — Gaussian bright-spot glare (camera flash / point source)
  gradient — diagonal gradient wash (window / ambient light reflection)

Usage:
    python benchmark_glare.py [--skip-run]   # --skip-run reuses existing preds
"""

import argparse
import json
import os
import sys
import re
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from PIL import Image
import Levenshtein

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

ROOT       = Path(__file__).parent.parent
DEMO_IMGS  = ROOT / 'omnidocbench_eval' / 'demo_data' / 'omnidocbench_demo' / 'images'
DEMO_GT_MD = ROOT / 'omnidocbench_eval' / 'demo_data' / 'omnidocbench_demo' / 'mds'
BENCH_DIR  = ROOT / 'preds' / 'glare_bench'


# ── Glare augmentation ────────────────────────────────────────────────────────

def _apply_specular(img: Image.Image, intensity: float = 0.65, seed: int = 0) -> Image.Image:
    """Gaussian bright-spot glare (camera flash / point-source reflection)."""
    rng = np.random.default_rng(seed)
    arr = np.array(img.convert('RGB'), dtype=np.float32)
    h, w = arr.shape[:2]
    cx = rng.choice([w * 0.25, w * 0.75])
    cy = rng.choice([h * 0.25, h * 0.75])
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    sigma = min(w, h) * 0.22
    blob = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    arr += blob[:, :, np.newaxis] * 255 * intensity
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _apply_gradient(img: Image.Image, intensity: float = 0.50) -> Image.Image:
    """Diagonal gradient wash (window / ambient-light reflection)."""
    arr = np.array(img.convert('RGB'), dtype=np.float32)
    h, w = arr.shape[:2]
    gx = np.linspace(0, 1, w)
    gy = np.linspace(0, 1, h)
    grad = (np.outer(gy, np.ones(w)) + np.outer(np.ones(h), gx)) / 2  # [H, W]
    arr += grad[:, :, np.newaxis] * 255 * intensity
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ── RAM tracker (main process + all child subprocesses) ───────────────────────

class PeakRAMTracker:
    def __init__(self, interval: float = 0.15):
        self._proc = psutil.Process(os.getpid())
        self._interval = interval
        self._peak_mb = 0.0
        self._running = False
        self._thread: threading.Thread | None = None

    def _total_rss_mb(self) -> float:
        try:
            rss = self._proc.memory_info().rss
            for child in self._proc.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return rss / 1e6
        except Exception:
            return 0.0

    def _loop(self):
        while self._running:
            v = self._total_rss_mb()
            if v > self._peak_mb:
                self._peak_mb = v
            time.sleep(self._interval)

    def start(self):
        self._peak_mb = self._total_rss_mb()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._running = False
        if self._thread:
            self._thread.join()
        return self._peak_mb


# ── Per-page timing: capture from stdout ─────────────────────────────────────

class TimingCapture:
    """Intercept PRISM's 'done in Xs' prints to get per-page timings."""
    _RE = re.compile(r'done in ([\d.]+)s')

    def __init__(self):
        self.page_times: list[float] = []
        self._orig_write = None

    def __enter__(self):
        self._orig_write = sys.stdout.write
        capture = self

        def _write(s):
            m = capture._RE.search(s)
            if m:
                capture.page_times.append(float(m.group(1)))
            return capture._orig_write(s)

        sys.stdout.write = _write
        return self

    def __exit__(self, *_):
        sys.stdout.write = self._orig_write


# ── NED helpers ───────────────────────────────────────────────────────────────

def _ned(a: str, b: str) -> float:
    a = re.sub(r'\s+', ' ', a.strip().lower())
    b = re.sub(r'\s+', ' ', b.strip().lower())
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    return Levenshtein.distance(a, b) / max(len(a), len(b))


def load_gt_texts(gt_md_dir: Path, stems: list[str]) -> dict[str, str]:
    out = {}
    for stem in stems:
        md = gt_md_dir / f'{stem}.md'
        if md.exists():
            out[stem] = md.read_text(encoding='utf-8')
    return out


# ── PRISM runner ─────────────────────────────────────────────────────────────

def run_prism(image_paths: list[str], pred_dir: str) -> tuple[float, float, list[float]]:
    """Run PRISM, return (total_s, peak_ram_mb, per_page_times)."""
    sys.path.insert(0, str(ROOT))
    from benchmarks.run_omnidocbench import _run_prism_on_images

    pred_path = Path(pred_dir)
    pred_path.mkdir(parents=True, exist_ok=True)

    tracker = PeakRAMTracker()
    with TimingCapture() as tc:
        tracker.start()
        t0 = time.perf_counter()
        _run_prism_on_images(image_paths, pred_dir)
        elapsed = time.perf_counter() - t0
        peak_ram = tracker.stop()

    return elapsed, peak_ram, tc.page_times


# ── Report helpers ────────────────────────────────────────────────────────────

def _fmt(v, fmt='.2f', suffix=''):
    if v is None:
        return '  —'
    return f'{v:{fmt}}{suffix}'


def print_report(conditions: list[dict]):
    labels = [c['label'] for c in conditions]
    w = max(len(l) for l in labels) + 2

    rows = [
        ('Pages processed',          [str(c['n_pages'])            for c in conditions]),
        ('Total time (s)',            [_fmt(c['total_s'])           for c in conditions]),
        ('Per-page latency (s)',      [_fmt(c['per_page_s'])        for c in conditions]),
        ('Min page time (s)',         [_fmt(c.get('min_t'))         for c in conditions]),
        ('Max page time (s)',         [_fmt(c.get('max_t'))         for c in conditions]),
        ('Peak RAM – all procs (MB)', [_fmt(c['peak_ram_mb'], '.0f') for c in conditions]),
        ('Chars extracted (mean/pg)', [_fmt(c.get('mean_chars'), '.0f') for c in conditions]),
        ('Mean NED vs GT',            [_fmt(c.get('mean_ned'), '.4f') for c in conditions]),
        ('Text accuracy (1-NED)',     [_fmt(c.get('accuracy'), '.1f', '%') for c in conditions]),
    ]

    col_w = 22
    header = f"{'Metric':<38}" + ''.join(f'{l:>{col_w}}' for l in labels)
    print()
    print('=' * (38 + col_w * len(labels)))
    print('PRISM Glare Benchmark')
    print('=' * (38 + col_w * len(labels)))
    print(header)
    print('-' * (38 + col_w * len(labels)))
    for metric, vals in rows:
        print(f'{metric:<38}' + ''.join(f'{v:>{col_w}}' for v in vals))
    print('=' * (38 + col_w * len(labels)))

    # Per-page detail table
    all_stems = conditions[0].get('stems', [])
    if all_stems:
        print()
        print('Per-page latency (seconds)')
        print(f"{'Page stem':<55}" + ''.join(f'{l:>{col_w}}' for l in labels))
        print('-' * (55 + col_w * len(labels)))
        for i, stem in enumerate(all_stems):
            row = f'{stem[:53]:<55}'
            for c in conditions:
                times = c.get('page_times', [])
                v = times[i] if i < len(times) else None
                row += f'{_fmt(v):>{col_w}}'
            print(row)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-run', action='store_true',
                    help='Skip PRISM inference; reuse existing preds for eval only')
    ap.add_argument('--intensity-specular', type=float, default=0.65)
    ap.add_argument('--intensity-gradient', type=float, default=0.50)
    args = ap.parse_args()

    # ── Prepare image lists ──────────────────────────────────────────────────
    clean_imgs = sorted(DEMO_IMGS.glob('*.jpg'))
    if not clean_imgs:
        print(f'[!] No demo images found in {DEMO_IMGS}')
        sys.exit(1)

    stems = [p.stem for p in clean_imgs]
    print(f'[*] {len(clean_imgs)} demo images found')

    # ── Generate glared images ───────────────────────────────────────────────
    spec_dir = BENCH_DIR / 'specular_images'
    grad_dir = BENCH_DIR / 'gradient_images'
    spec_dir.mkdir(parents=True, exist_ok=True)
    grad_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_run:
        print('[*] Generating glared images...')
        for i, p in enumerate(clean_imgs):
            img = Image.open(p)
            _apply_specular(img, args.intensity_specular, seed=i).save(spec_dir / p.name)
            _apply_gradient(img, args.intensity_gradient).save(grad_dir / p.name)
        print(f'    specular -> {spec_dir}')
        print(f'    gradient -> {grad_dir}')

    # ── Run PRISM on all three conditions ────────────────────────────────────
    conditions_cfg = [
        ('clean',    [str(p) for p in clean_imgs],                   BENCH_DIR / 'preds_clean'),
        ('specular', [str(spec_dir / p.name) for p in clean_imgs],   BENCH_DIR / 'preds_specular'),
        ('gradient', [str(grad_dir / p.name) for p in clean_imgs],   BENCH_DIR / 'preds_gradient'),
    ]

    conditions = []
    for label, paths, pred_dir in conditions_cfg:
        pred_dir.mkdir(parents=True, exist_ok=True)

        if args.skip_run:
            total_s, peak_ram, page_times = None, None, []
        else:
            print(f'\n{"="*60}')
            print(f'[*] Running PRISM — {label.upper()} ({len(paths)} pages)')
            print(f'{"="*60}')
            total_s, peak_ram, page_times = run_prism(paths, str(pred_dir))
            print(f'[+] {label}: {total_s:.1f}s total, {peak_ram:.0f} MB peak RAM')

        conditions.append({
            'label': label,
            'n_pages': len(paths),
            'pred_dir': pred_dir,
            'total_s': total_s,
            'per_page_s': (total_s / len(paths)) if total_s else None,
            'min_t': min(page_times) if page_times else None,
            'max_t': max(page_times) if page_times else None,
            'peak_ram_mb': peak_ram,
            'page_times': page_times,
            'stems': stems,
        })

    # ── Accuracy: NED vs GT markdown ─────────────────────────────────────────
    print('\n[*] Computing NED vs GT markdown...')
    gt_texts = load_gt_texts(DEMO_GT_MD, stems)

    for c in conditions:
        neds, char_counts = [], []
        for stem in stems:
            pred_md = c['pred_dir'] / f'{stem}.md'
            if pred_md.exists():
                pred = pred_md.read_text(encoding='utf-8')
                char_counts.append(len(pred.strip()))
                if stem in gt_texts:
                    neds.append(_ned(pred, gt_texts[stem]))
        c['mean_ned']   = sum(neds) / len(neds) if neds else None
        c['accuracy']   = (1 - c['mean_ned']) * 100 if c['mean_ned'] is not None else None
        c['mean_chars'] = sum(char_counts) / len(char_counts) if char_counts else None

    # ── Print report ─────────────────────────────────────────────────────────
    print_report(conditions)

    # ── Save JSON results ─────────────────────────────────────────────────────
    out = []
    for c in conditions:
        out.append({k: v for k, v in c.items() if k not in ('pred_dir', 'stems')})
    results_path = BENCH_DIR / 'glare_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\n[*] Results saved to {results_path}')


if __name__ == '__main__':
    main()
