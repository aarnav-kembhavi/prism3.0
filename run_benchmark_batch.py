"""
run_benchmark_batch.py
----------------------
Re-runs the PRISM pipeline on every benchmark image, collects per-page
latency + RAM, writes prism_tex files, then runs run_full_benchmark.py.
"""

import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

IMAGES_DIR    = Path("benchmark_results/temp_images")
GT_DIR        = Path("pdf2latex_dataset/dataset")
PRISM_TEX_DIR = Path("benchmark_results/prism_tex")
LATENCY_LOG   = Path("benchmark_results/latency_log.csv")

PRISM_TEX_DIR.mkdir(parents=True, exist_ok=True)

# Only process pages that have a GT
image_paths = sorted(
    [p for p in IMAGES_DIR.glob("*.png") if (GT_DIR / f"{p.stem}_gt.tex").exists()],
    key=lambda p: int(p.stem),
)
print(f"[*] {len(image_paths)} pages to process (GT matched)\n")

latency_rows = []

for idx, img_path in enumerate(image_paths, 1):
    page_id  = img_path.stem          # "1", "2", ...
    out_dir  = Path(f"{page_id}_output")
    tex_out  = out_dir / "main.tex"

    print(f"[{idx:>2}/{len(image_paths)}] {img_path.name}", flush=True)

    try:
        proc = subprocess.run(
            [sys.executable, "orchestrate.py", str(img_path), "--profile"],
            capture_output=True,
            timeout=300,
            encoding="utf-8",
            errors="replace",
        )
        stdout = proc.stdout
    except subprocess.TimeoutExpired:
        print("  [!] TIMEOUT — skipping")
        continue
    except Exception as e:
        print(f"  [!] ERROR: {e} — skipping")
        continue

    if not tex_out.exists():
        print(f"  [!] No output tex — skipping (rc={proc.returncode})")
        print(proc.stderr[-300:] if proc.stderr else "")
        continue

    # Copy tex to prism_tex
    dest = PRISM_TEX_DIR / f"{page_id}_prism.tex"
    shutil.copy(tex_out, dest)

    # Parse profiling table from stdout
    row = {"page_id": page_id}
    for stage, key in [
        ("Normalization",  "norm_sec"),
        ("YOLO",           "yolo_sec"),
        ("OCR",            "ocr_sec"),
        ("Math",           "math_sec"),
        ("Table",          "table_sec"),
    ]:
        m = re.search(rf"{stage}\s*[^\|]*\|\s*([\d\.]+)s", stdout)
        row[key] = float(m.group(1)) if m else 0.0

    m_total = re.search(r"TOTAL\s*\|\s*([\d\.]+)s\s*\|\s*([\d\.]+)\s*MB", stdout)
    row["total_sec"] = float(m_total.group(1)) if m_total else 0.0
    row["peak_mb"]   = float(m_total.group(2)) if m_total else 0.0

    latency_rows.append(row)
    print(f"  total={row['total_sec']:.1f}s  peak={row['peak_mb']:.0f}MB  -> {dest.name}")

    # Clean up output dir
    shutil.rmtree(out_dir, ignore_errors=True)

# Write latency log
fields = ["page_id","norm_sec","yolo_sec","ocr_sec","math_sec","table_sec","total_sec","peak_mb"]
with open(LATENCY_LOG, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(latency_rows)
print(f"\n[OK] Latency log -> {LATENCY_LOG}  ({len(latency_rows)} pages)")

# Run full benchmark
print("\n" + "="*60)
print("  Running full metric evaluation...")
print("="*60 + "\n")
subprocess.run([sys.executable, "run_full_benchmark.py"])
