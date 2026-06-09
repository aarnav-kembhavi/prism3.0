import os
import time
import json
import csv
import re
import statistics
import subprocess
import shutil
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

# Metrics
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from evaluation.normalizer import normalize_latex, split_math_and_text
from evaluation.eval import levenshtein_distance

# --- CONFIGURATION ---
DATASET_DIR = Path("pdf2latex_dataset/dataset")
RESULTS_DIR = Path("benchmark_results")
if RESULTS_DIR.exists():
    shutil.rmtree(RESULTS_DIR)
RESULTS_DIR.mkdir()

TEMP_IMG_DIR = RESULTS_DIR / "temp_images"
TEMP_IMG_DIR.mkdir()

OUTPUT_TEX_DIR = RESULTS_DIR / "prism_tex"
OUTPUT_TEX_DIR.mkdir()

# --- UTILS ---

def pdf_to_png(pdf_path, output_path):
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # PDF2LaTeX-102 is single page
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
    pix.save(output_path)
    doc.close()

def compute_bleu(pred, gt):
    bleu = BLEU(effective_order=True)
    try:
        return bleu.sentence_score(pred, [gt]).score
    except:
        return 0.0

def compute_rouge(pred, gt):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    try:
        scores = scorer.score(gt, pred)
        return scores['rougeL'].fmeasure * 100
    except:
        return 0.0

def compute_cer_wer(pred, gt):
    # CER
    dist = levenshtein_distance(pred, gt)
    cer = (dist / len(gt)) * 100 if len(gt) > 0 else 0
    
    # WER
    p_words = pred.split()
    g_words = gt.split()
    dist_w = levenshtein_distance(p_words, g_words)
    wer = (dist_w / len(g_words)) * 100 if len(g_words) > 0 else 0
    
    return cer, wer

def parse_orchestrate_output(output):
    """
    Extracts per-stage latency and memory from the summary table.
    """
    table_pattern = re.compile(r"^\s*([\w\s\(\)]+?)\s*\|\s*([\d\.]+)s\s*\|\s*([\d\.]+)\s*MB", re.MULTILINE)
    matches = table_pattern.findall(output)
    
    breakdown = {}
    for stage, latency, mem in matches:
        stage = stage.strip()
        breakdown[stage] = {
            "latency": float(latency),
            "memory": float(mem)
        }
    
    total_match = re.search(r"TOTAL\s*\|\s*([\d\.]+)s\s*\|\s*([\d\.]+)\s*MB", output)
    cpu_match = re.search(r"CPU \(Mean/Peak\)\s*\|\s*([\d\.]+)%\s*\|\s*([\d\.]+)%", output)
    
    return {
        "stages": breakdown,
        "total_latency": float(total_match.group(1)) if total_match else 0,
        "total_memory": float(total_match.group(2)) if total_match else 0,
        "cpu_mean": float(cpu_match.group(1)) if cpu_match else 0,
        "cpu_peak": float(cpu_match.group(2)) if cpu_match else 0
    }

# --- MAIN BENCHMARK LOOP ---

def run_benchmark(limit=None):
    all_page_metrics = []
    
    gt_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith("_gt.tex")], key=lambda x: int(x.split("_")[0]))
    if limit:
        gt_files = gt_files[:limit]
    
    print(f"[*] Starting benchmark on {len(gt_files)} pages...")
    
    for gt_fname in gt_files:
        base_id = gt_fname.split("_")[0]
        pdf_path = DATASET_DIR / f"{base_id}.pdf"
        gt_path = DATASET_DIR / gt_fname
        png_path = TEMP_IMG_DIR / f"{base_id}.png"
        
        print(f"[*] Processing Page {base_id}...")
        
        try:
            # 1. Convert PDF to PNG
            pdf_to_png(pdf_path, png_path)
            
            # 2. Run PRISM
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            cmd = ["python", "-u", "orchestrate.py", str(png_path), "--profile"]
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                                  timeout=300, env=env)
            
            if proc.returncode != 0:
                print(f"[!] Error on page {base_id}: {proc.stderr}")
                continue
                
            # 3. Collect Latency/Resource metrics
            perf = parse_orchestrate_output(proc.stdout)
            
            # 4. Read Generated LaTeX
            prism_out_dir = Path(f"{base_id}_output")
            prism_tex_path = prism_out_dir / "main.tex"
            
            if not prism_tex_path.exists():
                print(f"[!] main.tex not found for {base_id}")
                continue
                
            with open(prism_tex_path, "r", encoding='utf-8') as f:
                pred_latex = f.read()
            with open(gt_path, "r", encoding='utf-8') as f:
                gt_latex = f.read()
                
            # Store a copy
            shutil.copy(prism_tex_path, OUTPUT_TEX_DIR / f"{base_id}_prism.tex")
            
            # 5. Normalize and Evaluate
            # Accuracy Metrics: Character-level (spaces removed)
            pred_norm = normalize_latex(pred_latex, remove_spaces=True)
            gt_norm = normalize_latex(gt_latex, remove_spaces=True)
            
            # Accuracy Metrics: Word-level (spaces preserved)
            pred_word = normalize_latex(pred_latex, remove_spaces=False)
            gt_word = normalize_latex(gt_latex, remove_spaces=False)
            
            # Split (on character-level normalized)
            pred_math, pred_text = split_math_and_text(pred_norm)
            gt_math, gt_text = split_math_and_text(gt_norm)
            
            # Accuracy Metrics
            ed_total = levenshtein_distance(pred_norm, gt_norm)
            edr_total = 1.0 - (ed_total / len(gt_norm)) if len(gt_norm) > 0 else 1.0
            
            ed_math = levenshtein_distance(pred_math, gt_math)
            edr_math = 1.0 - (ed_math / len(gt_math)) if len(gt_math) > 0 else 1.0
            
            ed_text = levenshtein_distance(pred_text, gt_text)
            edr_text = 1.0 - (ed_text / len(gt_text)) if len(gt_text) > 0 else 1.0
            
            # CER/WER
            cer, _ = compute_cer_wer(pred_norm, gt_norm) # CER on characters
            _, wer = compute_cer_wer(pred_word, gt_word) # WER on words
            
            # BLEU/ROUGE on words
            bleu = compute_bleu(pred_word, gt_word)
            rouge = compute_rouge(pred_word, gt_word)
            
            # Forensic Math Log
            if len(all_page_metrics) < 5:
                print(f"\n--- Forensic Math Log (Page {base_id}) ---")
                print(f"PRED NORM (start): {pred_norm[:200]}...")
                print(f"HAS DOLLAR: {'$' in pred_norm}")
                print(f"PRED MATH: {pred_math[:200]}...")
                print(f"GT MATH:   {gt_math[:200]}...")
                print(f"EDR MATH:  {edr_math:.4f}")
                print("-" * 40)
            
            page_data = {
                "id": base_id,
                "overall_edr": edr_total,
                "math_edr": edr_math,
                "text_edr": edr_text,
                "edit_distance": ed_total,
                "bleu": bleu,
                "rouge": rouge,
                "cer": cer,
                "wer": wer,
                "latency": perf["total_latency"],
                "mem_peak": perf["total_memory"],
                "cpu_mean": perf["cpu_mean"],
                "ocr_latency": perf["stages"].get("OCR (Rapid)", {}).get("latency", 0),
                "layout_latency": perf["stages"].get("YOLO (ONNX)", {}).get("latency", 0),
                "math_latency": perf["stages"].get("Math (Texo)", {}).get("latency", 0),
                "table_latency": perf["stages"].get("Table (Solver)", {}).get("latency", 0),
                "gt_len": len(gt_norm)
            }
            all_page_metrics.append(page_data)
            
            # Cleanup page output to save space
            if prism_out_dir.exists():
                shutil.rmtree(prism_out_dir, ignore_errors=True)

        except Exception as e:
            print(f"[!] Exception on page {base_id}: {e}")

    # --- AGGREGATION ---
    if not all_page_metrics:
        print("[!] No metrics collected.")
        return

    summary = {
        "overall_edr": statistics.mean([p["overall_edr"] for p in all_page_metrics]),
        "math_edr": statistics.mean([p["math_edr"] for p in all_page_metrics]),
        "text_edr": statistics.mean([p["text_edr"] for p in all_page_metrics]),
        "avg_bleu": statistics.mean([p["bleu"] for p in all_page_metrics]),
        "avg_rouge": statistics.mean([p["rouge"] for p in all_page_metrics]),
        "avg_cer": statistics.mean([p["cer"] for p in all_page_metrics]),
        "avg_wer": statistics.mean([p["wer"] for p in all_page_metrics]),
        "avg_latency": statistics.mean([p["latency"] for p in all_page_metrics]),
        "peak_ram": max([p["mem_peak"] for p in all_page_metrics]),
        "avg_ram": statistics.mean([p["mem_peak"] for p in all_page_metrics]),
        "avg_cpu": statistics.mean([p["cpu_mean"] for p in all_page_metrics]),
        "avg_ocr_latency": statistics.mean([p["ocr_latency"] for p in all_page_metrics]),
        "avg_layout_latency": statistics.mean([p["layout_latency"] for p in all_page_metrics]),
        "avg_math_latency": statistics.mean([p["math_latency"] for p in all_page_metrics]),
        "avg_table_latency": statistics.mean([p["table_latency"] for p in all_page_metrics]),
    }
    
    # Save CSVs
    with open(RESULTS_DIR / "benchmark_results.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_page_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_page_metrics)
        
    # Generate Report
    report = f"""# PRISM vs PDF2LaTeX Benchmark Report

## Executive Summary
This report evaluates the PRISM Screen-to-LaTeX system against the PDF2LaTeX (Wang & Liu, 2020) dataset and baseline.

## Dataset Statistics
- **Number of pages:** {len(all_page_metrics)}
- **Total Ground Truth Characters:** {sum(p["gt_len"] for p in all_page_metrics)}

## Comparison Table

| Metric | PDF2LaTeX (Paper) | PRISM (Ours) | Improvement |
| ------ | ----------------- | ------------ | ----------- |
| Overall EDR | 81.1% | {summary['overall_edr']:.1%} | {((summary['overall_edr'] - 0.811) / 0.811):.1%} |
| BLEU-4 | 92.1* | {summary['avg_bleu']:.1f} | - |
| Avg. Latency | - | {summary['avg_latency']:.2f}s | - |
| Peak RAM | - | {summary['peak_ram']:.1f} MB | - |

*Note: PDF2LaTeX reported 92.1 BLEU for formula recognition specifically.*

## Performance Breakdown
- **Average Total Latency:** {summary['avg_latency']:.2f}s
- **Average Peak RAM:** {summary['avg_ram']:.1f} MB
- **Peak RAM (max page):** {summary['peak_ram']:.1f} MB
- **Average OCR Latency:** {summary['avg_ocr_latency']:.2f}s
- **Average Layout Latency:** {summary['avg_layout_latency']:.2f}s
- **Average Math Latency:** {summary['avg_math_latency']:.2f}s
- **Average Table Latency:** {summary['avg_table_latency']:.2f}s

## Failure Analysis
TBD

"""
    with open(RESULTS_DIR / "final_report.md", "w") as f:
        f.write(report)
        
    print("[✓] Benchmark complete. Results in benchmark_results/")

if __name__ == "__main__":
    # Full benchmark run
    run_benchmark()
