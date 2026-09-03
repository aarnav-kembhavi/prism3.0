# PRISM quantization sweep

Goal: shrink every ONNX graph in the stack as far as it goes **without wrecking
output quality**. The bar is not OmniDocBench score — it is that output stays
structurally correct and readable *versus the current fp32 output*.

Revert everything with one command:

```bash
git reset --hard fp32-baseline
```

## Layout

| Path | What |
|---|---|
| `eval_pages.json` | The frozen 30-page eval set. **Never regenerate.** |
| `eval_gt.json` | OmniDocBench GT subset for those 30 pages |
| `calib_pages.json` | 100 held-out calibration pages, disjoint from the eval set |
| `baseline/` | fp32 reference output (`*.md`), `perf.json` |
| `baseline_meta.json` | fp32 latency / peak RSS / per-graph on-disk sizes |
| `variants/<name>/` | one quantized variant: `*.md`, `meta.json`, `compare.json`, `worst_diffs.txt`, `graph_manifest.jsonl` |
| `quantize_log.json` | every quantization attempt, including the ones that failed |
| `results_table.md` | the per-variant results table |

## Selecting graphs at run time

`pipeline/quant_select.py` registers every graph and picks fp32 vs INT8 from the
environment. fp32 files are never overwritten; INT8 siblings carry an `_int8`
suffix.

```bash
PRISM_QUANT=texo_decoder,ppocr_rec        # only those two load INT8
PRISM_QUANT=all                           # every graph with an INT8 sibling
PRISM_QUANT=                              # pure fp32 (default)

PRISM_QUANT_PPDOCLAYOUT_V3=0              # force one graph back to fp32
PRISM_QUANT_PPDOCLAYOUT_V3=path/to/x.onnx # or point it at an explicit graph
```

A selected-but-missing INT8 graph falls back to fp32 with a printed warning
rather than dying, so a half-built sweep stays runnable.

### Proving what actually loaded

Model loads happen in worker **subprocesses** whose stdout the benchmark runner
does not capture, so a print cannot prove which graph a stage opened. Set
`PRISM_QUANT_MANIFEST=<file.jsonl>` and every process appends what it resolved,
including the calling module. `run_variant.py` sets this automatically and
records the result in `meta.json` under `graphs_actually_loaded`.

This is not paranoia. `pipeline/text_worker.py` builds its own model paths
(it replicates `models_interface`), so an early version of this sweep patched
only `models_interface` and measured the **fp32** OCR stack while believing it
was INT8 — the giveaway was similarity of exactly 1.00000 on all 30 pages.

## Rebuilding the quantized graphs

The `_int8` graphs are gitignored (~102 MB, deterministic to rebuild):

```bash
python quant/quantize.py texo_encoder   --mode dynamic
python quant/quantize.py texo_decoder   --mode dynamic --subgraphs
python quant/quantize.py ppocr_rec      --mode dynamic
python quant/quantize.py ppocr_det      --mode dynamic
python quant/quantize.py ppdoclayout_v3 --mode static --calib-n 100
python quant/quantize.py ppdoclayout_v3 --mode static --calib-n 100 --exclude-heads \
    --out models/ppdoclayoutv3/PP-DocLayoutV3_int8_heads.onnx
```

Every graph goes through `quant_pre_process` first. If it fails, the graph is
logged and skipped — never force-quantized.

### Graph-specific findings

- **`--subgraphs` (`EnableSubgraph`) is required for the Texo decoder.** The
  merged decoder is a single top-level `If` node holding all 27.5 MB of
  initializers, with every MatMul inside its two branches. Without the flag the
  quantizer never descends into them and the graph comes back *0.2% larger*.
- **`quant_pre_process` needs `skip_symbolic_shape` for the merged decoder and
  for PP-DocLayoutV3.** `symbolic_shape_infer` asserts on sequence/optional
  types inside `If` subgraphs. The fallback still runs ONNX shape inference and
  the optimizer; which mode was used is recorded per graph in
  `quantize_log.json` as `preprocess_mode`.
- **Static QDQ must be restricted to `Conv`/`MatMul`/`Gemm`.** Letting the
  quantizer touch elementwise `Add` produces `QLinearAdd` nodes that ORT
  refuses to load (`Scale and Zero-point must be a scalar`). This is not a
  per-channel problem — it reproduces with `per_channel=False`. Conv/MatMul/Gemm
  hold all the weight anyway.
- **SLANet-plus cannot be quantized with this toolchain.** `quant_pre_process`
  succeeds, but `quantize_dynamic` fails inside its own shape inference with
  `Inferred shape and existing shape differ in dimension 0: (256) vs (768)`.
  Logged and skipped; the fp32 graph is untouched.

## Running a variant

```bash
python quant/run_variant.py --name texo_decoder --quant texo_decoder
python quant/compare.py     --variant texo_decoder
python quant/report.py
```

`compare.py` scores each page two independent ways: normalized Levenshtein
similarity against the fp32 output, and structural checks that edit distance
cannot see — balanced LaTeX environments and delimiters, unchanged table/row/cell
counts, non-empty output, and runaway repetition in formula decode. The last one
matters because a degenerate INT8 decoder that starts looping barely moves edit
distance on a short formula inside a long page.

## The verification probe

PP-OCRv6 det is used twice: for OCR text boxes, and again at 640 px inside
`normalization/verified.py` as the probe that accepts or rejects every
normalization proposal. Quantizing it moves a **decision gate**, not just box
coordinates.

That gate is invisible on the normal eval path: `benchmarks/run_omnidocbench.py`
sets `PRISM_NORM_STRICT=1`, and `normalization/pipeline.py` disables the whole
verified path when it is set. `quant/probe_rate.py` therefore measures it in a
dedicated harness with `PRISM_NORM_STRICT=0`, running normalization only, once
per det variant:

```bash
python quant/probe_rate.py     # -> quant/probe_rate.json
```

## Reading the latency numbers

Median s/page on this machine varies run to run for an identical configuration
(the fp32 decoder variant measured 7.03 s and 9.49 s on two runs of the same
graphs). `variants/fp32_repeat/` is a full fp32 re-run kept as the noise floor —
compare any latency or similarity delta against it before believing it.
