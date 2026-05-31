# =============================================================================
# config.py — Central configuration for Stage 2 pipeline
# =============================================================================

CONFIG = {
    "lang":                    ["en"],
    "indent_threshold_px":     20,
    "row_merge_tolerance_px":  10,
    "output_json":             "stage2_output.json",
    "gpu":                     False,
    "preprocess_deskew":       True,
    "preprocess_denoise":      True,    # NLM for text; bilateral for tables (see preprocess.py)
    "preprocess_contrast":     True,
    "preprocess_contrast_factor": 2.2,
    "preprocess_binarize":     False,
    "preprocess_sharpen":      True,
    "tatr_confidence":         0.15,    # Low threshold — let NMS handle duplicates
    "ocr_confidence":          0.2,
    # SLANet DISABLED: PPStructure (old API) is not available in PaddleOCR v3+.
    # Enabling it causes a 10-30s import failure + retry loop on every table.
    # Use TATR + PaddleOCR instead (reliable, faster).
    "use_slanet":              False,
    # table_upscale REMOVED: upscaling to 2x before TATR quadrupled pixel count
    # and degraded structure detection (TATR is trained at native document DPI).
    # Cell-crop OCR still upscales small cells internally when h < 120px.
    "table_min_col_width_ratio": 0.04,
    "table_merge_dollar_cols": True,
    "table_merge_currency_prefix": True,
    "ppstructure_lang":        "en",
    "ppstructure_text_recognition_model_name": "ch_PP-OCRv4_rec",
    "ocr_backend":             "paddle",
}

# TATR pretrained model — Microsoft, trained on PubTables-1M
TATR_MODEL = "microsoft/table-transformer-structure-recognition"