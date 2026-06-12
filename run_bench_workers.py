"""
run_bench_workers.py
--------------------
26-page benchmark using subprocess workers (MathOCRWorker + TextOCRWorker).
Replicates run_bench_26.py logic exactly, but routes math/text through workers
so the main process never imports PyTorch — measuring true main-process peak RSS.
"""
import gc, os, shutil, statistics, time
from pathlib import Path
import numpy as np
import psutil

from math_worker_onnx import MathOCRWorkerOnnx as MathOCRWorker
from text_worker import TextOCRWorker

from normalization import normalize_image_pil
from normalization.modality import CaptureModality
from models_interface import (
    get_yolo_model, unload_yolo,
    run_table_extraction_batched,
)
from layout_utils import (
    apply_semantic_reading_order, xyxy_to_pil_crop,
    detect_column_count, split_detections_by_column,
)
from latex_builder import wrap_content, assemble_document, save_tex
from detection_postprocess import postprocess_detections
from evaluation.normalizer import normalize_latex, split_math_and_text
from run_full_benchmark import compute_edr

GT_DIR     = Path("pdf2latex_dataset/dataset")
IMAGES_DIR = Path("benchmark_results/temp_images")
OUT_BASE   = Path("bench_tmp_workers")
YOLO_MODEL = "yolov11n-doclaynet.onnx"

TEXT_CLASSES  = {"Text","Title","Section-header","Caption","Footnote","Page-footer","Page-header","List-item"}
MATH_CLASSES  = {"Formula"}
TABLE_CLASSES = {"Table"}
IMAGE_CLASSES = {"Picture"}
LIST_ITEM     = "List-item"

FORMULA_PAD        = 12
HEADER_SUPPRESS_H  = 0.12   # top 12% of page
HEADER_H_FRAC      = 0.065
HEADER_W_FRAC      = 0.25

all_imgs = sorted(
    [p for p in IMAGES_DIR.glob("*.png") if (GT_DIR/f"{p.stem}_gt.tex").exists()],
    key=lambda p: int(p.stem),
)[:26]

process = psutil.Process(os.getpid())


def _is_logo(c):
    a = np.array(c.convert("RGB"), dtype=np.float32)
    return float(np.mean(a.mean(2) < 230)) < 0.15 and float(a.std()) > 8.0


def _route_and_extract(dets, figures_dir, math_w, text_w, is_ss, math_ctr):
    ti  = [i for i,d in enumerate(dets) if d["class_name"] in TEXT_CLASSES]
    mi  = [i for i,d in enumerate(dets) if d["class_name"] in MATH_CLASSES]
    tai = [i for i,d in enumerate(dets) if d["class_name"] in TABLE_CLASSES]

    if mi:
        results, math_ctr = math_w.run_math_batch(
            [dets[i]["crop"] for i in mi], figures_dir, math_ctr)
        for idx, raw in zip(mi, results): dets[idx]["raw_content"] = raw

    if ti:
        texts = text_w.run_text_batch([dets[i]["crop"] for i in ti], is_screenshot=is_ss)
        for idx, txt in zip(ti, texts): dets[idx]["raw_content"] = txt

    if tai:
        t_res = run_table_extraction_batched([dets[i]["crop"] for i in tai])
        for idx, raw in zip(tai, t_res): dets[idx]["raw_content"] = raw

    parts, li, fc = [], set(), 0
    for det in dets:
        cn = det["class_name"]; raw = det.get("raw_content", "")
        if cn in TEXT_CLASSES | MATH_CLASSES:
            if cn == LIST_ITEM: li.add(len(parts))
            parts.append(wrap_content(cn, raw))
        elif cn in TABLE_CLASSES and raw:
            parts.append(wrap_content(cn, raw))
        elif cn in IMAGE_CLASSES:
            fc += 1; fname = f"figure_{fc:03d}.png"
            det["crop"].save(os.path.join(figures_dir, fname))
            parts.append(wrap_content("Picture", fname))
    return parts, li, fc, math_ctr


def run_page(img_path, out_dir, yolo, math_w, text_w):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    assets  = out_dir / "assets"
    figures = assets / "figures"
    for d in [out_dir, assets, figures]: d.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    img_norm, img_fid, mod = normalize_image_pil(str(img_path))
    is_ss = (mod.modality == CaptureModality.SCREENSHOT)
    norm_path = str(assets / "normalized.png"); img_norm.save(norm_path)
    w, h = img_norm.width, img_norm.height

    # YOLO detection
    res = yolo(norm_path, verbose=False)
    dets = []
    for box in res[0].boxes:
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        cn = res[0].names[int(box.cls[0].item())]
        crop = xyxy_to_pil_crop(img_fid if cn in IMAGE_CLASSES else img_norm, [x1,y1,x2,y2])
        if cn == "Page-header":
            fc = xyxy_to_pil_crop(img_fid, [x1,y1,x2,y2])
            if _is_logo(fc): cn = "Picture"; crop = fc
        dets.append({"bbox":[x1,y1,x2,y2],"class_name":cn,"confidence":float(box.conf[0]),"crop":crop})
    dets = postprocess_detections(dets, w, h)

    # Header suppression
    suppress_y = h * HEADER_SUPPRESS_H
    dets = [d for d in dets if not (
        d["class_name"] in {"Section-header","Page-header"} and d["bbox"][3] <= suppress_y)]

    # Header logo injection
    header_box = [w*(1-HEADER_W_FRAC), 0, w, h*HEADER_H_FRAC]
    if not any(d["class_name"]=="Picture" and d["bbox"][0]>=header_box[0]
               and d["bbox"][3]<=header_box[3] for d in dets):
        hx1,hy1,hx2,hy2 = [int(v) for v in header_box]
        hcrop = xyxy_to_pil_crop(img_fid, [hx1,hy1,hx2,hy2])
        if hcrop.width > 20:
            dets.insert(0, {"bbox":[hx1,hy1,hx2,hy2],"class_name":"Picture",
                            "confidence":1.0,"crop":hcrop,"is_header_logo":True})

    # Formula padding + re-crop
    for det in dets:
        x1,y1,x2,y2 = det["bbox"]
        if det["class_name"] in MATH_CLASSES:
            x1=max(0,x1-FORMULA_PAD); y1=max(0,y1-FORMULA_PAD)
            x2=min(w,x2+FORMULA_PAD); y2=min(h,y2+FORMULA_PAD)
        det["crop"] = xyxy_to_pil_crop(
            img_fid if det["class_name"] in IMAGE_CLASSES else img_norm,
            [x1,y1,x2,y2])
    del img_norm, img_fid; gc.collect()

    figs_str = str(figures)
    header_dets = [d for d in dets if d.get("is_header_logo")]
    body_dets   = [d for d in dets if not d.get("is_header_logo")]
    header_logo = None
    if header_dets:
        header_logo = "assets/figure_header_logo.png"
        header_dets[0]["crop"].save(str(out_dir / header_logo))

    col_count = detect_column_count(body_dets, w)
    math_ctr  = 0

    if col_count == 2:
        full_dets, left_dets, right_dets = split_detections_by_column(
            body_dets, w, h, use_dag=True)
        full_p, full_li, _, math_ctr = _route_and_extract(
            full_dets, figs_str, math_w, text_w, is_ss, math_ctr)
        left_p, left_li, _, math_ctr = _route_and_extract(
            left_dets, figs_str, math_w, text_w, is_ss, math_ctr)
        right_p, right_li, _, math_ctr = _route_and_extract(
            right_dets, figs_str, math_w, text_w, is_ss, math_ctr)
        doc = assemble_document(full_p, full_li, True,
                                left_p, left_li, right_p, right_li, header_logo)
    else:
        body_sorted = apply_semantic_reading_order(body_dets, w, h)
        body_p, body_li, _, _ = _route_and_extract(
            body_sorted, figs_str, math_w, text_w, is_ss, math_ctr)
        doc = assemble_document(body_p, body_li, False, header_logo=header_logo)

    tex = out_dir / "main.tex"; save_tex(doc, str(tex))

    elapsed = time.perf_counter() - t0
    peak_mb = process.memory_info().rss / 1024**2
    pred = normalize_latex(tex.read_text(encoding="utf-8",errors="ignore"), remove_spaces=True)
    gt   = normalize_latex((GT_DIR/f"{img_path.stem}_gt.tex").read_text(encoding="utf-8",errors="ignore"), remove_spaces=True)
    pm, pt = split_math_and_text(pred); gm, gt_ = split_math_and_text(gt)
    return {"pid": img_path.stem, "overall": compute_edr(pred,gt),
            "math": compute_edr(pm,gm), "text": compute_edr(pt,gt_),
            "sec": elapsed, "mb": peak_mb}


if __name__ == '__main__':
    print("[*] Starting subprocess workers...")
    math_w = MathOCRWorker(); math_w.start()
    text_w = TextOCRWorker(); text_w.start()
    print(f"[*] Workers ready. Main process RAM: {process.memory_info().rss/1024**2:.0f} MB")

    yolo = get_yolo_model(YOLO_MODEL)
    print(f"[*] YOLO loaded. Main process RAM: {process.memory_info().rss/1024**2:.0f} MB")

    rows = []
    for i, img_path in enumerate(all_imgs, 1):
        print(f"[{i:>2}/26] {img_path.stem}", flush=True)
        r = run_page(img_path, OUT_BASE / img_path.stem, yolo, math_w, text_w)
        rows.append(r)
        print(f"  {r['sec']:.1f}s  {r['mb']:.0f}MB  EDR={r['overall']:.1%}  text={r['text']:.1%}  math={r['math']:.1%}")
        gc.collect()

    math_w.stop(); text_w.stop()
    unload_yolo(); gc.collect()

    print(f"\n{'='*56}")
    print("  PRISM + Subprocess Workers  —  26 PDF2LaTeX pages")
    print(f"{'='*56}")
    print(f"  Pages       : {len(rows)}/26")
    print(f"  Overall EDR : {statistics.mean(r['overall'] for r in rows):.1%}  (median {statistics.median(r['overall'] for r in rows):.1%})")
    print(f"  Text EDR    : {statistics.mean(r['text'] for r in rows):.1%}")
    print(f"  Math EDR    : {statistics.mean(r['math'] for r in rows):.1%}")
    print(f"  Avg latency : {statistics.mean(r['sec'] for r in rows):.1f}s  (median {statistics.median(r['sec'] for r in rows):.1f}s)")
    print(f"  Avg peak RAM (main process): {statistics.mean(r['mb'] for r in rows):.0f} MB")
    print(f"  Max peak RAM (main process): {max(r['mb'] for r in rows):.0f} MB")
    print(f"{'='*56}")
