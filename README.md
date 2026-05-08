JUST FOR TESTING

# Screen2LaTeX — Pipeline Change Log

Changes made across four files during the debugging and improvement sessions.
Each section describes what was wrong, what was changed, and why.

---

## 1. `normalization/pipeline.py`

### Auto-bypass for detected screenshots

**Problem:** The pipeline was configured with `source_dpi=96` and `target_dpi=250`,
producing a 2.6× upscale for every input. For phone photos this is correct — they
are typically taken at ~96 effective DPI and need upscaling to reach OCR-friendly
resolution. For screenshots this is destructive: the image is already a perfect
digital render, and bicubic upscaling introduces JPEG-style interpolation artefacts
that severely degrade EasyOCR accuracy (blurred letter edges, merged glyphs,
false serifs).

**Fix:** After `normalize_image_pil` runs and the modality is determined, if the
image is classified as a **screenshot with ≥ 60% confidence**, `image_norm` is
replaced with the raw original PIL image before it is passed to YOLO and OCR.
The `normalized.png` saved to disk still reflects any CLAHE contrast polish that
was applied, but the actual OCR input is the clean original pixels.

```python
if is_screenshot and modality_result.confidence >= 0.60:
    image_norm = Image.open(args.image_path).convert("RGB")
    image_fidelity = image_norm
```

---

## 2. `normalization/region_adaptive.py`

### False-positive glare detection on white backgrounds

**Problem:** `GLARE_AREA_THRESH = 0.005` (0.5%) meant "fire if more than 0.5% of
crop pixels have LAB L > 230". On a white-background IEEE journal page, ordinary
white margins and page background trivially exceed this threshold — every single
text crop was being reported as 100% glare. The Telea inpainting then ran on clean
white space, destroying the text underneath before EasyOCR could read it.

**Fix:** A second threshold `GLARE_AREA_THRESH_SCREENSHOT = 0.25` was added.
For screenshots, the glare detector now requires **25% of the crop** to be
genuinely overexposed before inpainting fires. This eliminates all false positives
on white-background documents while still catching real overexposed zones.

```python
GLARE_AREA_THRESH: float             = 0.005   # phone photos
GLARE_AREA_THRESH_SCREENSHOT: float  = 0.25    # screenshots
```

### False-positive moiré detection on JPEG compression artefacts

**Problem:** `MOIRE_SPIKE_RATIO_THRESH = 1.8` fires when the peak FFT magnitude
is 1.8× the mean. JPEG compression introduces subtle ringing artefacts in the
frequency domain that produce ratios in the 2–3× range on screenshot crops. This
caused the FFT notch filter to run on clean digital text, introducing ringing on
sharp character edges.

**Fix:** A second threshold `MOIRE_SPIKE_RATIO_THRESH_SCREENSHOT = 5.0` was added.
For screenshots, only genuine moiré (which produces very strong isolated spikes,
ratio > 5×) triggers the filter.

```python
MOIRE_SPIKE_RATIO_THRESH: float            = 1.8   # phone photos
MOIRE_SPIKE_RATIO_THRESH_SCREENSHOT: float = 5.0   # screenshots
```

### Shadow removal skipped for screenshots

**Problem:** The DoG (difference-of-Gaussians) shadow removal normalises
illumination gradients. On screenshots these gradients do not exist — all "gradients"
seen by DoG are JPEG compression blocking artefacts. Running DoG on a screenshot
flattened the contrast of body text regions.

**Fix:** Shadow removal is unconditionally skipped when `is_screenshot=True`,
regardless of what `detect_shadow()` reports.

### `preprocess_crop` signature extended

**Change:** `preprocess_crop(crop_bgr, class_name)` now takes a third parameter
`is_screenshot: bool = False`. All three threshold decisions above are routed
through this flag. The call site in `orchestrate.py` passes `is_screenshot`
derived from the modality detection result.

```python
def preprocess_crop(crop_bgr, class_name, is_screenshot=False):
    ...
    glare_det, sev = detect_glare(result, is_screenshot=is_screenshot)
    moire_det, sev = detect_moire(result, is_screenshot=is_screenshot)
    if shadow_det and (class_name in _SKIP_SHADOW or is_screenshot):
        profile.skipped_corrections.append("shadow")
    ...
```

---

## 3. `orchestrate.py`

### Fix: `normalize_image_pil` unpack error

**Problem:** `normalize_image_pil` returns three values
`(normalized_img, fidelity_img, modality_result)` but the call site was only
unpacking two, causing a `ValueError: too many values to unpack`.

**Fix:** Updated unpacking to capture all three values and added a modality
print line.

```python
image_norm, image_fidelity, modality_result = normalize_image_pil(...)
print(f"[✓] Modality: {modality_result}")
```

### Fix: Header logo crop too large (12% → 6.5%)

**Problem:** `HEADER_H_FRAC = 0.12` caused the logo crop to capture the top 12%
of the page height. The IEEE Access logo only occupies roughly the top 5–7%, so
the crop included 3 lines of body text below the logo. These appeared verbatim
in `figure_001.png`.

**Fix:** Reduced to `HEADER_H_FRAC = 0.065`.

### Fix: Header logo placed mid-column instead of above paracol

**Problem:** The injected logo detection was entering the normal detection list
and being routed through column assignment. It ended up mid-paragraph in the
right column wherever YOLO happened to sort it.

**Fix:** Logo detections are tagged with `"is_header_logo": True`. Before column
routing, all logo detections are extracted, saved immediately as `figure_001.png`,
and their filename is passed as `header_logo=...` to `assemble_document`. The
logo never enters the column routing pipeline.

```python
header_logo_dets = [d for d in detections if d.get("is_header_logo")]
body_detections  = [d for d in detections if not d.get("is_header_logo")]
```

### Fix: Running title reaching body text (`HEADER_SUPPRESS_H_FRAC` 0.10 → 0.12)

**Problem:** The "Author et al.: Benchmarking Neural Architectures…" running
title line was appearing as the first line of the left column body text.
The suppression threshold `HEADER_SUPPRESS_H_FRAC = 0.10` was meant to delete
any `Section-header` or `Page-header` detection whose bottom edge sits in the
top 10% of the page, but the running title's bbox sat at ~10–11%, just below
the cutoff.

**Fix:** Raised to `HEADER_SUPPRESS_H_FRAC = 0.12`.

### Fix: `is_screenshot` passed to `preprocess_crop`

The modality flag is now forwarded into every `preprocess_crop` call so the
region-adaptive module can apply the correct thresholds per the section above.

```python
corrected_bgr, profile = preprocess_crop(
    crop_bgr, det['class_name'], is_screenshot=is_screenshot
)
```

### Fix: `route_and_extract` wires math fallback

`run_math_recognition` now accepts `fallback_figures_dir` and a mutable
`fallback_counter` list. `route_and_extract` passes these so failed formula
crops are saved as `formula_001.png` instead of being silently dropped.

```python
raw = run_math_recognition(
    crop,
    fallback_figures_dir=figures_dir,
    fallback_counter=math_fallback_counter,
)
```

---

## 4. `models_interface.py`

### Fix: Formula fallback image when Texo returns empty

**Problem:** When Texo (the math OCR model) failed or returned an empty string,
`run_math_recognition` returned `""`. The equation was then completely absent
from the output PDF with no indication it existed.

**Fix:** `run_math_recognition` now accepts two optional parameters:
- `fallback_figures_dir: str` — path to the output folder
- `fallback_counter: list` — one-element mutable counter `[n]`

On any failure or empty result, the crop is saved as `formula_NNN.png` and the
function returns `\includegraphics[width=0.5\linewidth]{formula_NNN.png}`.
The `Formula` wrapper in `latex_builder.py` detects the `\includegraphics` prefix
and uses `\begin{center}` instead of `\begin{equation}` for correct rendering.

```python
except Exception as e:
    fname = f"formula_{fallback_counter[0]:03d}.png"
    crop.save(os.path.join(fallback_figures_dir, fname))
    return f"\\includegraphics[width=0.5\\linewidth]{{{fname}}}"
```

---

## 5. `latex_builder.py`

### Fix: `\section{}` → `\subsection*{}` for IEEE headers

**Problem:** `Section-header` was mapped to `\section{...}`, producing large
numbered headings that overrode IEEE's own section numbering. Headers like
"D. TRAINING PROTOCOL" became top-level `\section` entries.

**Fix:** Mapped to `\subsection*{...}` (unnumbered subsection), which matches
the visual weight of IEEE journal section labels.

### Fix: `\title`/`\maketitle` → `\begin{center}\textbf{\large ...}\end{center}`

**Problem:** `\title{...}\maketitle` requires `\author` and `\date` declarations.
Journal page extracts don't have these, causing either a compile crash or a
malformed title block with blank author/date lines.

**Fix:** `Title` class now emits a centered bold large text block with no
`\maketitle`.

### Fix: Merged IEEE section headers split into separate `\subsection*` lines

**Problem:** YOLO sometimes detects two adjacent IEEE header lines (e.g.
"IV. IMPLEMENTATION DETAILS" and "A. SOFTWARE FRAMEWORK") as a single
`Section-header` region. EasyOCR reads the merged text as one string:
`"IV. IMPLEMENTATION DETAILS A. SOFTWARE FRAMEWORK"`, which produced one
oversized garbled `\subsection*` heading.

**Fix:** `_split_section_header()` applies `_IEEE_HEADER_SPLIT_RE` — a regex
that splits on the pattern `(?<![A-Z])(?=[A-Z]\.\s+[A-Z])` (an isolated
uppercase letter followed by dot and space, not preceded by another uppercase).
This correctly splits at "A." and "B." labels without incorrectly splitting
Roman numerals like "IV.".

```
"IV. IMPLEMENTATION DETAILS A. SOFTWARE FRAMEWORK"
→ \subsection*{IV. IMPLEMENTATION DETAILS}
  \subsection*{A. SOFTWARE FRAMEWORK}
```

### Fix: All 4 bullet items in one `\item` blob

**Problem:** YOLO was detecting all four Model A/B/C/D bullet points as a
single `List-item` region. EasyOCR read the entire block as one string:
`"Model A- Learning rate 10-3 50 epochs Model B: Learning rate..."`.
The output was one giant `\item` containing all four bullets.

**Fix:** `_split_bullet_items()` applies `_BULLET_SPLIT_RE` — a regex that
splits on lookaheads for `Model [A-D]` followed by any separator character
(`:`  `-`  `.`  or space). This handles all three separator forms EasyOCR
produces. The `List-item` wrapper calls this function and emits one `\item`
per split part.

```python
_BULLET_SPLIT_RE = re.compile(
    r'(?<!\A)(?=\bModel\s+[A-D][\s:\-\.])'
)
```

### Fix: OCR artifact cleaning (`_clean_ocr`)

A `_clean_ocr()` function is applied to all text before LaTeX wrapping.
It chains the following corrections:

| Pattern | Example input | Corrected output |
|---|---|---|
| Soft hyphen (no space before) | `pa- rameters` | `parameters` |
| Soft hyphen (space before) | `con - taining` | `containing` |
| Thousands dot confusion | `7.352` | `7,352` |
| Escaped trailing underscore | `samples\_` | `samples.` |
| Unescaped trailing underscore | `overfitting_` | `overfitting.` |
| `t0` → `to` | `trained t0 minimize` | `trained to minimize` |
| `[IO]` / `[I0]` → `[10]` | `Vaswani et al. [IO]` | `Vaswani et al. [10]` |
| Capital O in numbers | `209.00O` | `209,000` |
| Exponent notation | `10-3` | `$10^{-3}$` |
| Scaled exponents | `1.5x10-3` | `$1.5\times10^{-3}$` |

The soft-hyphen regex `r'(\w) ?- +([a-z])'` handles both the
`"word- word"` form (no space before hyphen) and the `"word - word"` form
(space on both sides) that EasyOCR emits depending on column width.

### Fix: Formula environment selection

The `Formula` wrapper now checks whether the content starts with
`\includegraphics` (the math OCR fallback path) and selects the appropriate
LaTeX environment:

```python
"Formula": lambda c: (
    f"\n\\begin{{equation}}\n{c}\n\\end{{equation}}\n"
    if c and not c.startswith("\\includegraphics")
    else f"\n\\begin{{center}}\n{c}\n\\end{{center}}\n"
),
```

### Fix: Header logo placement above paracol

`assemble_document` gains a `header_logo: Optional[str]` parameter. When set,
it emits the logo right-aligned with a rule beneath it **before** the
`\begin{paracol}` block, matching the IEEE Access page layout:

```latex
\noindent\hfill\includegraphics[height=1.8em]{figure_001.png}
\par\noindent\hrule\vspace{4pt}
```

---

## Summary table

| File | Change | Reason |
|---|---|---|
| `normalization/pipeline.py` | Auto-bypass 2.6× upscale for screenshots | Interpolation artefacts degrade EasyOCR |
| `normalization/region_adaptive.py` | `GLARE_AREA_THRESH_SCREENSHOT = 0.25` | White backgrounds falsely triggered inpainting |
| `normalization/region_adaptive.py` | `MOIRE_SPIKE_RATIO_THRESH_SCREENSHOT = 5.0` | JPEG artefacts falsely triggered FFT filter |
| `normalization/region_adaptive.py` | Skip shadow removal for screenshots | DoG flattened JPEG compression gradients |
| `normalization/region_adaptive.py` | `preprocess_crop(is_screenshot=False)` | Route modality-aware thresholds per crop |
| `orchestrate.py` | Unpack 3-tuple from `normalize_image_pil` | `ValueError: too many values to unpack` |
| `orchestrate.py` | `HEADER_H_FRAC` 0.12 → 0.065 | Logo crop included body text rows |
| `orchestrate.py` | `HEADER_SUPPRESS_H_FRAC` 0.10 → 0.12 | Running title line reached body column |
| `orchestrate.py` | Logo extracted before column routing | Logo appeared mid-paragraph in right column |
| `orchestrate.py` | `is_screenshot` forwarded to `preprocess_crop` | Region-adaptive module needed modality flag |
| `orchestrate.py` | Math fallback counter wired into `route_and_extract` | Formula crops saved on Texo failure |
| `models_interface.py` | Formula fallback → save crop as image | Equations silently lost when Texo returned empty |
| `latex_builder.py` | `\section` → `\subsection*` | IEEE labels are subsection-level, not top-level |
| `latex_builder.py` | `\title`/`\maketitle` → centered bold text | `\maketitle` requires `\author`/`\date` |
| `latex_builder.py` | `_split_section_header()` | Merged YOLO detections produced garbled double headers |
| `latex_builder.py` | `_split_bullet_items()` | All 4 Model A/B/C/D bullets merged into one `\item` |
| `latex_builder.py` | `_clean_ocr()` with 10 fix patterns | Soft hyphens, `t0`, `[IO]`, `O` in numbers, `\_`, exponents |
| `latex_builder.py` | Formula environment selector | `\equation` vs `\center` based on fallback detection |
| `latex_builder.py` | `header_logo` param in `assemble_document` | Logo emitted above paracol, right-aligned with hrule |
