import os
import time
import json
import csv
import subprocess
import shutil
from pathlib import Path
from PIL import Image
import statistics

# Metrics
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from evaluation.normalizer import normalize_latex
from evaluation.eval import levenshtein_distance

# --- CONFIGURATION ---
ROBUSTNESS_DIR = Path("benchmark_results/robustness")
if ROBUSTNESS_DIR.exists():
    shutil.rmtree(ROBUSTNESS_DIR)
ROBUSTNESS_DIR.mkdir(parents=True)

# Define Test Cases
test_images = {
    "clean_screenshot": "image.png",
    "phone_photo_glare": "glare2.jpeg",
    "phone_photo_shadow": "image3.jpeg",
    "low_contrast": "glared.jpeg",
}

rotations = [5, 15, 30, 90, 180]
base_for_rotation = "image.png"

# --- UTILS ---

def parse_perf(output):
    total_match = re.search(r"TOTAL\s*\|\s*([\d\.]+)s\s*\|\s*([\d\.]+)\s*MB", output)
    return {
        "latency": float(total_match.group(1)) if total_match else 0,
        "memory": float(total_match.group(2)) if total_match else 0
    }

import re

# --- EXECUTION ---

def run_robustness():
    results = []
    
    # 1. Standard categories
    for category, img_path in test_images.items():
        print(f"[*] Robustness: {category} ({img_path})...")
        cmd = ["python", "orchestrate.py", img_path, "--profile"]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        stem = Path(img_path).stem
        out_dir = Path(f"{stem}_output")
        tex_path = out_dir / "main.tex"
        
        perf = parse_perf(proc.stdout)
        
        success = tex_path.exists()
        results.append({
            "category": category,
            "image": img_path,
            "success": success,
            "latency": perf["latency"],
            "memory": perf["memory"]
        })
        
        if success:
            shutil.copy(tex_path, ROBUSTNESS_DIR / f"{category}.tex")
            shutil.rmtree(out_dir)

    # 2. Rotations
    base_img = Image.open(base_for_rotation)
    for angle in rotations:
        print(f"[*] Robustness: Rotation {angle} deg...")
        rot_path = ROBUSTNESS_DIR / f"rotated_{angle}.png"
        base_img.rotate(-angle, expand=True, fillcolor="white").save(rot_path)
        
        cmd = ["python", "orchestrate.py", str(rot_path), "--profile"]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        stem = rot_path.stem
        out_dir = Path(f"{stem}_output")
        tex_path = out_dir / "main.tex"
        
        perf = parse_perf(proc.stdout)
        success = tex_path.exists()
        
        results.append({
            "category": f"rotation_{angle}",
            "image": str(rot_path),
            "success": success,
            "latency": perf["latency"],
            "memory": perf["memory"]
        })
        
        if success:
            shutil.copy(tex_path, ROBUSTNESS_DIR / f"rotation_{angle}.tex")
            if out_dir.exists(): shutil.rmtree(out_dir)

    # Save results
    with open(ROBUSTNESS_DIR / "robustness_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("[✓] Robustness tests complete.")

if __name__ == "__main__":
    run_robustness()
