"""Run MinerU (pipeline backend, CPU) on the 20-page subset with RAM/latency
instrumentation. Collects per-image markdown into a pred dir.
Run with the venvs/mineru_cpu python (for MetricsTracker deps psutil)."""
import os, sys, json, time, shutil, subprocess, glob, threading
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "compare"))
os.chdir(ROOT)
import psutil

SUBSET = "benchmarks/compare/compare20_subset.json"
IMAGES = "data/omnidocbench/images"
IN_DIR = "benchmarks/compare/mineru_in_full"
OUT_DIR = "benchmarks/compare/mineru_out_full"
PRED_DIR = "benchmarks/compare/preds_mineru"
# MinerU CPU baseline venv (see SETUP.md). Windows layout first, POSIX fallback.
_MINERU_VENV = ROOT / "venvs" / "mineru_cpu"
MINERU = next(
    (str(p) for p in (_MINERU_VENV / "Scripts" / "mineru.exe",
                      _MINERU_VENV / "bin" / "mineru")
     if p.exists()),
    str(_MINERU_VENV / "Scripts" / "mineru.exe"))

for d in (IN_DIR, OUT_DIR, PRED_DIR):
    os.makedirs(d, exist_ok=True)

d = json.load(open(SUBSET, encoding='utf-8'))
stems = []
for p in d:
    name = p['page_info']['image_path']
    src = os.path.join(IMAGES, name)
    if os.path.exists(src):
        shutil.copy(src, IN_DIR)
        stems.append(os.path.splitext(name)[0])
print(f"[mineru] {len(stems)} images")

# peak RSS sampler over the whole process tree
peak = [0.0]; running = [True]
def sample():
    me = psutil.Process(os.getpid())
    while running[0]:
        tot = 0
        try:
            for pr in [me] + me.children(recursive=True):
                try: tot += pr.memory_info().rss
                except Exception: pass
        except Exception: pass
        peak[0] = max(peak[0], tot / 1024 / 1024)
        time.sleep(0.2)
th = threading.Thread(target=sample, daemon=True); th.start()

env = dict(os.environ, MINERU_MODEL_SOURCE='huggingface', MINERU_DEVICE_MODE='cpu',
           HF_HUB_DISABLE_SYMLINKS_WARNING='1')
t0 = time.perf_counter()
subprocess.run([MINERU, '-p', IN_DIR, '-o', OUT_DIR, '-b', 'pipeline', '-m', 'ocr'],
               env=env, check=False)
wall = time.perf_counter() - t0
running[0] = False; th.join(timeout=2)

# collect markdown
n = 0
for stem in stems:
    hits = glob.glob(os.path.join(OUT_DIR, stem, "**", f"{stem}.md"), recursive=True)
    md = open(hits[0], encoding='utf-8').read() if hits else ''
    open(os.path.join(PRED_DIR, f"{stem}.md"), 'w', encoding='utf-8').write(md)
    if md: n += 1

eff = {"model": "mineru", "n_pages": len(stems), "device": "cpu",
       "peak_rss_mb": round(peak[0], 1), "latency_total_s": round(wall, 1),
       "latency_mean_s": round(wall / max(len(stems), 1), 2),
       "latency_median_s": round(wall / max(len(stems), 1), 2), "peak_vram_mb": None}
json.dump(eff, open(os.path.join(PRED_DIR, "_efficiency.json"), 'w'), indent=2)
print(f"[mineru] DONE {n}/{len(stems)} md, peak_rss={eff['peak_rss_mb']}MB, "
      f"total={wall:.0f}s, ~{eff['latency_mean_s']}s/page")
