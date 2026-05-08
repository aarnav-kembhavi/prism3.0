# Screen2LaTeX — Samsung Prism Pipeline

End-to-end pipeline that converts document images (phone photos, screenshots, scans) into compilable LaTeX. Handles real-world artifacts like glare, shadows, moiré, and perspective distortion.

## Pipeline Stages

```
Input Image
    │
    ├── Stage 1: Image Normalization (optional)
    │   ├── White balance correction (gray world)
    │   ├── Geometric rectification (multi-strategy: morph gradient / Hough lines / Canny)
    │   ├── Shadow removal (difference-of-Gaussians)
    │   ├── Glare inpainting (LAB threshold + cv2.inpaint)
    │   ├── Moiré removal (FFT notch filter)
    │   ├── Contrast normalization (CLAHE)
    │   └── Smart DPI resize
    │
    ├── Stage 2: Layout Analysis
    │   ├── YOLO detection (DocLayNet, 11 classes)
    │   ├── Post-processing (NMS, overlap resolution, box refinement)
    │   └── Column detection (single vs two-column)
    │
    ├── Stage 3: Content Extraction
    │   ├── Text / Title / Caption / Headers → EasyOCR
    │   ├── Formula → rapid_latex_ocr
    │   ├── Table → (stub — model TBD)
    │   └── Picture → saved as figure_NNN.png
    │
    └── Stage 4: LaTeX Assembly
        ├── Class-aware LaTeX wrapping
        ├── Reading order sorting
        ├── Single / two-column (paracol) layout
        └── Output: main.tex + figures
```

## Setup

**Requirements:** Python 3.13+, [uv](https://docs.astral.sh/uv/)

```bash
# Clone and install
git clone <repo-url>
cd Samsung-Prism-pipeline
uv sync
```

This installs all dependencies: `ultralytics`, `easyocr`, `rapid-latex-ocr`, `opencv-python`, `scipy`, `pillow`, `paddleocr`, `paddlepaddle`, `numpy`, `psutil`.

## Usage

### With Normalization (phone photos, camera captures)

Use this for images taken with a phone camera that may have glare, shadows, perspective distortion, or moiré patterns.

```bash
uv run python orchestrate.py path/to/photo.jpeg
```

The normalization pipeline will:
- Detect and correct perspective distortion
- Remove glare spots via inpainting
- Even out shadows
- Clean moiré patterns
- Normalize contrast

### Without Normalization (clean inputs)

Use `--skip-normalize` for clean PDF renders, digital screenshots, or scanned documents that don't have camera artifacts.

```bash
uv run python orchestrate.py path/to/clean_image.png --skip-normalize
```

### Additional Options

```bash
# Adjust YOLO confidence threshold (default: 0.25)
uv run python orchestrate.py image.png --conf 0.35

# Custom DPI settings (default: source=96, target=250)
uv run python orchestrate.py image.jpeg --target-dpi 300 --source-dpi 72

# Profile CPU, Memory, and Latency (outputs to terminal and profiling_report.csv)
uv run python orchestrate.py image.jpeg --profile
```

### When to use which mode

| Input Type | Command |
|-----------|---------|
| Phone photo of a document | `uv run python orchestrate.py photo.jpeg` |
| Screenshot of a PDF | `uv run python orchestrate.py screenshot.png --skip-normalize` |
| Scanned document | `uv run python orchestrate.py scan.png --skip-normalize` |
| Photo with heavy glare | `uv run python orchestrate.py glared.jpeg` |

## Output

Running the pipeline creates a folder `<image_stem>_output/`:

```
image_output/
├── main.tex           ← Compilable LaTeX document
├── normalized.png     ← Cleaned image (if normalization ran)
├── figure_001.png     ← Cropped picture regions
├── figure_002.png
└── ...
```

**To compile:** Upload the entire folder to [Overleaf](https://overleaf.com) and compile `main.tex`. Image paths are filename-only, so Overleaf resolves them in the same directory.

## Evaluation

Compare predicted LaTeX against ground-truth using Levenshtein edit distance with PDF2LaTeX normalization.

```python
from evaluation import evaluate_page, evaluate_dataset

# Single page
result = evaluate_page(predicted_latex, groundtruth_latex)
print(f"Edit Distance Rate: {result['edit_distance_rate']:.1%}")
print(f"Math EDR: {result['math_edit_dist_rate']:.1%}")
print(f"Text EDR: {result['text_edit_dist_rate']:.1%}")

# Batch (folder of .tex files)
summary = evaluate_dataset("predictions/", "groundtruth/", "results.json")
```

## Project Structure

```
├── orchestrate.py              ← CLI entry point
├── normalization/
│   ├── pipeline.py             ← 7-step normalization + Fidelity Twin logic
│   ├── geometric.py            ← Multi-strategy document detection
│   └── frequency_filter.py     ← Glare, shadow, moiré, white balance
├── models_interface.py         ← OCR model interfaces (EasyOCR, rapid_latex_ocr)
├── layout_utils.py             ← Reading order, column detection, cropping
├── latex_builder.py            ← YOLO class → LaTeX mapping, document assembly
├── detection_postprocess.py    ← NMS, overlap resolution, box refinement
├── evaluation/
│   ├── eval.py                 ← Levenshtein EDR scoring
│   ├── normalizer.py           ← PDF2LaTeX 6-rule normalization
│   └── profiler.py             ← Background psutil resource monitor
├── yolov11n-doclaynet.pt       ← Pretrained YOLO model (DocLayNet)
└── pyproject.toml              ← Dependencies
```

## Models Used

| Task | Model | Notes |
|------|-------|-------|
| Layout Detection | YOLOv11n-DocLayNet | 11 document classes, CPU-friendly |
| Text OCR | EasyOCR | Lightweight, English |
| Math OCR | rapid_latex_ocr | Converts formula images to LaTeX |
| Table OCR | *(stub)* | Placeholder — needs SLANet/TFLOP |

## Normalization: Previous vs Current

| Step | Previous | Current |
|------|----------|---------|
| Document detection | Single Canny + contour (failed on glare) | 3 strategies: morph gradient → Hough lines → Canny |
| Glare removal | CLAHE only (redistributes contrast) | LAB threshold + inpainting (fills missing data) |
| Shadow removal | ❌ None | DoG illumination normalization |
| White balance | ❌ None | Gray world algorithm |
| DPI resize | Always 2.6× upscale | Smart resize (caps high-res inputs) |
| Figure Extraction | Cropped from highly distorted B&W tensor | **Hybrid Crop:** Extracts natural colors via perfectly-mapped Fidelity matrix |
| Profiling | ❌ None | Asynchronous CPU, RAM, & Latency monitoring via `psutil` thread |
