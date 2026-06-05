import subprocess
import os
import re
from pathlib import Path
import shutil
import json

images = [
    "glare2.jpeg",
    "glared.jpeg",
    "image3.jpeg",
    "image.png",
    "image2.png",
    "image4.png",
    "Texo/TechnoSelection/test_img/多行公式2.jpg",
    "Texo/TechnoSelection/test_img/单行公式.png",
    "Texo/TechnoSelection/test_img/单行公式2.png",
    "Texo/TechnoSelection/test_img/复杂表格.png",
    "Texo/TechnoSelection/test_img/多行公式.png"
]

final_folder = Path("final_folder")
if final_folder.exists():
    shutil.rmtree(final_folder)
final_folder.mkdir()

results = []

def parse_metrics(output, image_name):
    metrics = {"image": image_name}
    
    # Parse the summary table from orchestrate.py
    table_pattern = re.compile(r"^\s*([\w\s\(\)]+?)\s*\|\s*([\d\.]+)s\s*\|\s*([\d\.]+)\s*MB", re.MULTILINE)
    matches = table_pattern.findall(output)
    
    breakdown = {}
    for stage, latency, mem in matches:
        stage = stage.strip()
        breakdown[stage] = {
            "latency": float(latency),
            "memory": float(mem)
        }
    metrics["breakdown"] = breakdown
    
    # Parse total metrics from BackgroundProfiler
    total_match = re.search(r"TOTAL\s*\|\s*([\d\.]+)s\s*\|\s*([\d\.]+)\s*MB", output)
    if total_match:
        metrics["total_latency"] = float(total_match.group(1))
        metrics["total_memory_peak"] = float(total_match.group(2))

    # Parse CPU metrics
    cpu_match = re.search(r"CPU \(Mean/Peak\)\s*\|\s*([\d\.]+)%\s*\|\s*([\d\.]+)%", output)
    if cpu_match:
        metrics["cpu_mean"] = float(cpu_match.group(1))
        metrics["cpu_peak"] = float(cpu_match.group(2))
    
    return metrics

for img_path in images:
    print(f"[*] Processing {img_path}...")
    img_name = Path(img_path).name
    img_stem = Path(img_path).stem
    
    cmd = ["python", "orchestrate.py", img_path, "--profile"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', timeout=300)
        output = proc.stdout
        if proc.returncode != 0:
            print(f"[!] Error processing {img_path}: {proc.stderr}")
            continue
        
        m = parse_metrics(output, img_name)
        results.append(m)
        
        # Move output dir to final_folder
        out_dir = Path(f"{img_stem}_output")
        if out_dir.exists():
            dest = final_folder / img_stem
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(out_dir, dest)
            print(f"  [✓] Moved results to {dest}")
        
    except Exception as e:
        print(f"[!] Failed {img_path}: {e}")

# Save metrics to a file
with open("final_folder/metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n[✓] Batch processing complete.")
