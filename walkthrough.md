# Screen2LaTeX Orchestration Pipeline — Walkthrough

## What Was Built

Four Python files forming the complete orchestration layer for the Screen2LaTeX pipeline:

| File | Purpose |
|------|---------|
| [models_interface.py](file:///d:/DEVELOPMENT/Samsung%20Prism/Samsung-Prism_YOLO_orchestration/models_interface.py) | Stub functions for OCR, math recognition, and table extraction |
| [layout_utils.py](file:///d:/DEVELOPMENT/Samsung%20Prism/Samsung-Prism_YOLO_orchestration/layout_utils.py) | Bbox sorting (reading order) and image cropping |
| [latex_builder.py](file:///d:/DEVELOPMENT/Samsung%20Prism/Samsung-Prism_YOLO_orchestration/latex_builder.py) | YOLO class → LaTeX wrapping + full document assembly |
| [orchestrate.py](file:///d:/DEVELOPMENT/Samsung%20Prism/Samsung-Prism_YOLO_orchestration/orchestrate.py) | CLI entry point orchestrating the full pipeline |

## Pipeline Flow

```
Input image → YOLO detection (11 DocLayNet classes)
            → Sort by reading order (top→bottom, left→right)
            → Route each region to specialist model stub
            → Wrap output with LaTeX markup
            → Assemble compilable .tex document
            → Save to <stem>_output/ folder (tex + figure crops)
```

## Verification

- **Dependencies**: `ultralytics` and `pillow` confirmed installed via `uv add`
- **Import check**: All four modules import independently with no circular dependencies

## Usage

```bash
python orchestrate.py path/to/document_image.png
```

Output: `<image_stem>_output/` folder containing `main.tex` + any `figure_*.png` crops. Upload the entire folder to Overleaf to compile.
