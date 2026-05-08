# Pipeline Overview

Here is an overview of the core files and folders in the Screen2LaTeX pipeline:

### Core Python Files
*   **`orchestrate.py`**: The CLI entry point that runs the main pipeline. It loads the YOLO model, performs layout detection, routes cropped regions to specialist models (Text, Math, Table, Image), and orchestrates the final LaTeX document compilation.
*   **`models_interface.py`**: The interface for heavy, downstream specialist models. It lazy-loads and manages inference for EasyOCR (text), RapidLaTeXOCR (math/formulas), and PaddleOCR (tables). It also handles escaping LaTeX special characters.
*   **`latex_builder.py`**: The dedicated LaTeX generation module. It maps YOLO class labels to proper LaTeX wrappers (e.g., `\begin{equation}`, `\begin{itemize}`) and assembles the recognized text blocks into a fully compilable `.tex` document.
*   **`layout_utils.py`**: Utility functions for bounding box geometry. It calculates the correct document reading order (sorting by Top-Bottom, Left-Right) and detects/handles single-column vs. multi-column page layouts.
*   **`detection_postprocess.py`**: A cleanup module for raw YOLO layout predictions. It applies confidence filtering, class-aware Non-Maximum Suppression (NMS), and hierarchical overlap resolution to prevent text/region duplication.

### Folders
*   **`evaluation/`**: The testing and metric framework. Features scripts like `eval.py` to calculate the Levenshtein edit distance between output `.tex` files and ground truth, and `profiler.py` for tracking CPU/memory usage and inference latency.
*   **`normalization/`**: The Stage 1 image preprocessing pipeline. It prepares raw phone photos and screenshots for YOLO by running tasks like structural rectification (`geometric.py`), glare removal and contrast balancing (`frequency_filter.py`), and smart DPI scaling (`pipeline.py`).
*   **`text-table-latex/`**: A standalone or alternative Stage 2 sub-pipeline focused specifically on text and table resolution from screenshots. It uses EasyOCR and Microsoft's Table Transformer (TATR) and features advanced formatting utilities (`pipeline.py`) and specialized screenshot noise pre-processing (`preprocess.py` like deskewing and denoising).
