# WACV 2027 — Task 2 results

**Scope:** Task 1 (ablation residual) skipped per instruction. This covers Task 2
(no-correction arm on the 44 real captures) and the two ALSO items.

**Build:** HEAD `d91c6b1` / tag v21, working tree unmodified. No `.tex` file was
touched. New files are additive only:
`scripts/rebuttal/task2_capture_arms.py`, `scripts/rebuttal/task2_score.py`,
`results/rebuttal/task2_captures_3arm.json`, `results/rebuttal/task2_summary.json`.

---

## 0. Environment (ALSO, item 2) — CONFIRMED

| | value |
|---|---|
| CPU model string | `11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz` |
| cores / logical processors | 8 physical / 16 logical |
| onnxruntime (PRISM env `venvs/gpu`, used for every PRISM number) | **1.20.1** |
| onnxruntime (`venvs/mineru_cpu`, MinerU baseline env only) | 1.27.0 |

`venvs/docling_rebuttal`, `venvs/ppocr`, `venvs/smol` have no onnxruntime.
The CPU string matches the `i7-11800H` already named in the `tab:perf_threads`
caption — CONFIRMED, no edit needed there.

---

## 1. TASK 2 — three arms on the 44 uncontrolled real captures

### 1.1 What ran

132 page runs (44 captures × 3 arms), 593 s wall, one process, `venvs/gpu`.
Driver: `scripts/rebuttal/task2_capture_arms.py`.

| arm | definition |
|---|---|
| `none` | corrections disabled: deskew + 1800 px shorter-side cap only — byte-for-byte the input the other two arms receive |
| `open` | forced open-loop stack, ungated: white\_balance → rectify → glare → shadow → clahe(if gray std < 45) |
| `verified` | identical stack, every step through `normalization.verified.verified_apply`, `_ACCEPT_GAIN = 1.02` (submitted default, unmodified) |

Full OCR text + per-line boxes/confidences are persisted per arm in
`results/rebuttal/task2_captures_3arm.json`, so any later metric can be computed
offline without re-running the stack.

### 1.2 FAILED — the requested metric cannot be computed

> **The 44 captures have no ground truth. Both requested quantities are blocked.**

- **`mean block edit distance` per arm — FAILED (no GT).**
- **`pages damaged = edit distance worsened by >0.02 vs raw` — FAILED (no GT).**

`test_images/real/defects/defects-images/` contains 44 images (`1.jpg` … `45.png`,
no `3`) and nothing else but a `.DS_Store`. I swept every `.json` / `.jsonl` /
`.txt` / `.md` in the repo for files keyed by those filenames; the only six hits
are `results/rebuttal/modality_audit.json`, `scratchpad_runs/probe_gate/probe_log{,_v2}.jsonl`,
`pages_meta{,_v2}.jsonl`, and `TASK1B_REPORT.md` — all logs, none a transcription.
There is no GT text and no GT block geometry, so there is nothing to take an edit
distance *against*.

This is why every prior analysis of this set (`TASK1B_REPORT.md`, Phase 2, Phase 3)
uses an OCR **char-yield** proxy rather than edit distance. `tab:verified` (the
none/open/verified table you refer to as Tab S8) is a *different* set — the 40
synthetic-degradation benchmark pages, which do have GT. **The 44 captures were
never edit-distance scored and cannot be without a transcription pass.**

No substitute reference was fabricated. Below is what the 132 runs actually
measure.

### 1.3 Per-arm results — 44 captures, paired bootstrap (10k, seed 0)

Same bootstrap protocol as `scratchpad_runs/probe_gate/bootstrap_cis.py`: one set
of resampled page indices reused across all three arms.

| arm | mean OCR chars/page [95% CI] | mean lines | mean conf | mean yield ratio vs `none` [95% CI] |
|---|---|---|---|---|
| `none` | 3810.2 [3410.2, 4202.8] | 104.4 | 0.9525 | 1.0000 (by construction) |
| `open` | 3520.5 [3073.8, 3943.2] | 98.0 | 0.9152 | **0.9050** [0.8340, 0.9597] |
| `verified` | 3794.2 [3397.1, 4185.7] | 104.3 | 0.9507 | **0.9963** [0.9894, 1.0020] |

Paired differences (`*` = CI excludes zero):

| contrast | Δ chars/page [95% CI] | Δ yield ratio [95% CI] |
|---|---|---|
| `open` − `none` | −289.70 [−476.09, −138.52] `*` | −0.0950 [−0.1660, −0.0403] `*` |
| `verified` − `none` | −15.91 [−40.18, **+3.34**] | −0.0037 [−0.0106, **+0.0020**] |
| `verified` − `open` | +273.80 [+122.93, +455.23] `*` | +0.0912 [+0.0371, +0.1601] `*` |

**Reading:** open-loop correction costs a statistically clear ~9.5 % of OCR char
yield on real captures; the gated stack is statistically indistinguishable from
doing nothing (CI straddles zero); the gap between gated and open-loop is clear
and in the gate's favour. This is the *damage-filter* claim, and it holds — it is
**not** a rescue claim (see §1.5).

### 1.4 Damage counts

Criterion is char yield, **not** edit distance. The >10 % row is the criterion
already used in Sec 4.6 / Phase 3; the >2 % row is included only as the nearest
available analog to your ">0.02" rule and is a different quantity.

| criterion (`none` > 200 chars) | `none` | `open` | `verified` | prevented by gate | induced by gate |
|---|---|---|---|---|---|
| **>10 % char loss vs `none`** | 0/44 | **9/44** | **1/44** | **9 of 9** | 1 (`11`) |
| >2 % char loss vs `none` | 0/44 | 19/44 | 3/44 | 17 of 19 | 1 (`16`) |

- `open` damaged at >10 %: `1, 5, 6, 9, 13, 14, 15, 24, 25` — **identical set and
  count to `phase3_threshold.md` and `TASK1B_REPORT.md` v2. CONFIRMED.**
- `verified` damaged: `11` only — **CONFIRMED** (the known residual; 3891 → 3481
  chars, −10.5 %, CLAHE accepted at ratio 1.047).
- Worst open-loop failures reproduce exactly: `1.jpg` 2864 → 114 chars (−96 %),
  `5.jpeg` 1513 → **0** chars (total collapse), `6.jpeg` 1417 → 825 (−42 %).
  `verified` ships all three byte-identical to `none`. **CONFIRMED.**

### 1.5 CHANGED — three Sec 4.6 claims do not survive

Gate decisions in my run reproduce `probe_log_v2.jsonl` exactly (41 accepted /
175 not accepted over 216 offers, 81.0 % rejection), so these are disagreements
with the *prose*, not with the build.

**(a) "the probe accepted shadow flattening on the 8 pages that pixel-statistics
routing had misclassified as clean" → the number is 10, and shadow was accepted
on 13 pages overall. CHANGED.**

Shadow accepted on 13/44: `8, 16, 17, 19, 24, 27, 29, 30, 31, 32, 33, 34, 39`.
Intersecting with the 25 captures `modality_audit.json` records as routed past
corrections gives **10**: `8, 16, 17, 27, 29, 30, 31, 32, 33, 34`. The other three
(`19, 24, 39`) were routed to the camera branch correctly, so they do not belong
in that clause either way.

**(b) "a $7\times$ probe-score drop on one hand-shadow page" → the measured worst
drop is $11\times$. CHANGED.**

Minimum accepted-vs-offered score ratio over the capture group is **0.0912** on
`18.png` (shadow flatten, rejected) = a **10.97×** drop. Next worst: `1.jpg`
shadow 0.3523 (2.84×), `12.png` shadow 0.3801 (2.63×). No proposal in the log
sits near 1/7 ≈ 0.143. Accepted median ratio 1.0372 (min 1.0203, max 2.1319);
rejected median 1.0002.

**(c) "end-to-end, previously empty outputs (receipt-class captures) became fully
readable" → NOT SUPPORTED by the three-arm data. CHANGED — this is the one I'd
flag hardest before submission.**

Across all 44 captures the **maximum** `verified`/`none` char-yield ratio is
**1.060** (`8.png`, 2857 → 3027). There is no page on which gating turns an empty
or near-empty output into a readable one:

- `4.jpeg` is the only empty page and is empty in **all three** arms (0 / 0 / 0
  chars) — it is 258×195 px and below the 450 px probe-trust floor.
- `5.jpeg`, the closest thing to the described case, runs the other way: `none`
  1513 chars, `open` **0**, `verified` 1513. The gate *preserved* a readable page
  that open-loop destroyed; it did not rescue an unreadable one.

This matches the caveat already recorded in `TASK1B_REPORT.md` ("On no capture
does verified improve full-res OCR char yield by >10 % — the gate's measurable
value on captures is damage prevention"). The paper sentence claims a rescue the
data do not show. The measurable, defensible claim on this set is **damage
prevention**, which is strong (9/9) and now has CIs.

### 1.6 Gate behaviour (context, CONFIRMED)

| step | accepted | not accepted | outcomes |
|---|---|---|---|
| white\_balance | 1 | 43 | 40 rejected, 3 low-res guard |
| rectify | 6 | 38 | 33 no-op, 2 rejected, 3 guard |
| glare | 1 | 43 | 30 no-op, 10 rejected, 3 guard |
| shadow | 13 | 31 | 28 rejected, 3 guard |
| CLAHE | 20 | 20 | 17 rejected, 3 guard |
| **total** | **41** | **175** | **81.0 % rejection** |

Matches `phase3_threshold.md`'s 19.0 % capture acceptance rate exactly.
`verified` output is bit-identical to `none` on 15/44 captures; `open` on 0/44.

---

## 2. ALSO, item 1 — Table 3 spread — CHANGED (the premise is wrong)

**Table 3 = `tab:cpu_frontier`** (`paper/sec/4_experiments.tex:81`, identical at
`paper_overleaf/sec/4_experiments.tex:81`).

**It does not say "mean of two runs", and it is not a mean of two runs — it is a
single run per system.** The caption says "same 20 pages, same machine, isolated
runs, 8-thread budget". Provenance: `docs/paperresults.md` §5 (local measurement
2026-07-02, PRISM pre-v10) ← `benchmarks/compare/RESULTS.md`, which states
"isolated runs (one system at a time)". There is one measurement per cell.
**No spread exists to report, and the logs are not "gone" — there was never a
second run for these numbers.** Nothing was re-run.

The "mean of two runs" phrasing you remember is real but belongs to a *different,
later* table: `scratchpad_runs/frontier/frontier_table.md`, a v21 rebuild on 20 EN
pages (seed 42), which does say "mean of two runs for latency/RAM" and **does**
carry per-run spread. Its numbers are not the submitted ones:

| system | RAM median-of-2 (GB) | RAM per-run [min, max] | s/pg median-of-2 | s/pg per-run [min, max] |
|---|---|---|---|---|
| PRISM (v21) | 1.961 | [1.958, 1.963] | 15.62 | [15.433, 15.809] |
| MinerU-pipeline 3.4.4 | 3.119 | [3.106, 3.131] | 18.05 | [16.991, 19.103] |
| PP-StructureV3 (server) | 5.736 | [5.505, 5.966] | 176.36 | [164.87, 187.85] |
| PP-StructureV3 (mobile) | 4.600 | [4.374, 4.826] | 18.67 | [17.71, 19.62] |
| SmolDocling-256M | 1.805 | [1.796, 1.814] | 129.67 | [126.87, 132.47] |
| GraniteDocling-258M | 1.918 | [1.912, 1.925] | 83.58 | [79.66, 87.49] |

Accuracy cells there are single-run ("Accuracy is from run 1 (deterministic)"), so
they have no spread either.

**Separate discrepancy noticed while tracing, no re-run:** submitted Table 3 gives
PP-StructureV3 (server) **8.0 GB**; the source `benchmarks/compare/RESULTS.md`
records **8.2 GB**. `docs/paperresults.md` §5 already carries the 8.0 figure, so
the rounding was introduced upstream of the `.tex`. Flagging, not changing.

---

## 3. Diff-ready LaTeX edits — NOT APPLIED, awaiting approval

All three edits are to the same sentence, in both copies:
`paper/sec/4_experiments.tex:250` and `paper_overleaf/sec/4_experiments.tex:231`.
No other line needs to change for Task 2.

### Edit 1 (required) — replace the unsupported Sec 4.6 capture sentence

```diff
-Those cases are real but extreme --- on the 44 uncontrolled defect captures (hand shadows, tungsten lamps, page curl, deep corner shadows), the probe accepted shadow flattening on the 8 pages that pixel-statistics routing had misclassified as clean and \emph{rejected} it precisely where the divide erased strokes in dense paragraphs (a $7\times$ probe-score drop on one hand-shadow page); end-to-end, previously empty outputs (receipt-class captures) became fully readable. Open-loop stacks buy those rescues by taxing every clean page; the closed loop gets them for free.
+Those cases are real but extreme. On the 44 uncontrolled defect captures (hand shadows, tungsten lamps, page curl, deep corner shadows) the probe accepted shadow flattening on 13 pages --- 10 of them pages that pixel-statistics routing had misclassified as clean --- and \emph{rejected} it precisely where the divide erased strokes in dense paragraphs (an $11\times$ probe-score drop on the worst hand-shadow page). Measured across all three arms, the gate's value on real captures is \emph{damage prevention}, not rescue: open-loop correction costs $9.5\%$ of OCR character yield on average ($-0.095$ mean yield ratio, 95\% paired-bootstrap CI $[-0.166, -0.040]$ over the 44 pages), and damages 9 of 44 captures by more than $10\%$ --- one of them (\texttt{5.jpeg}) to zero recognized characters. The gated stack prevents \emph{all nine} while remaining statistically indistinguishable from applying no correction at all ($-0.004$ mean yield ratio, CI $[-0.011, +0.002]$); it ships output bit-identical to the uncorrected page on 15 of 44 captures and rejects $81\%$ of the corrections it is offered. Open-loop stacks pay that tax on every page to buy corrections these captures did not need; the closed loop declines to pay it.
```

Rationale per clause: `8 → 13 (10 misclassified)` §1.5(a); `$7\times$ → $11\times$`
§1.5(b); the "previously empty outputs became fully readable" clause is deleted
because §1.5(c) shows no such page exists; everything added is measured in §1.3–1.6.

### Edit 2 (optional, defensive) — make the `tab:verified` caption state its scope

Pre-empts "why no edit distance on the real captures?" from a reviewer.

```diff
-\caption{Synthetic-defect study: mean per-block OCR edit distance (lower is better) on 40 text-heavy benchmark pages (20 EN / 20 ZH) under harsh photometric degradations with valid ground truth. \emph{none} = no correction, \emph{open} = open-loop camera stack, \emph{verified} = probe-gated stack.}
+\caption{Synthetic-defect study: mean per-block OCR edit distance (lower is better) on 40 text-heavy benchmark pages (20 EN / 20 ZH) under harsh photometric degradations with valid ground truth. \emph{none} = no correction, \emph{open} = open-loop camera stack, \emph{verified} = probe-gated stack. Degradations are synthetic because the metric requires ground-truth text-block geometry; the 44 uncontrolled captures are uncalibrated and are therefore reported by recognition yield rather than edit distance.}
```

### Edit 3 (optional) — Table 3 caption, if you want the single-run status explicit

```diff
-\caption{Controlled CPU head-to-head: same 20 pages, same machine, isolated runs, 8-thread budget. PRISM row predates its current accuracy (conservative). $^{*}$format mismatch; $^{\dagger}$config artifact.}
+\caption{Controlled CPU head-to-head: same 20 pages, same machine, one isolated run per system, 8-thread budget. PRISM row predates its current accuracy (conservative). $^{*}$format mismatch; $^{\dagger}$config artifact.}
```

---

## 4. Flag summary

| # | item | flag |
|---|---|---|
| 1 | mean block edit distance per arm, 44 captures | **FAILED** — no GT exists; TODO left standing |
| 2 | pages damaged, edit distance >0.02 vs raw | **FAILED** — same cause |
| 3 | 132 page runs, 3 arms, completed | **CONFIRMED** (593 s) |
| 4 | open-loop damages 9/44, ids `1,5,6,9,13,14,15,24,25` | **CONFIRMED** |
| 5 | verified damages 1/44 (`11`) | **CONFIRMED** |
| 6 | gate prevents 9 of 9 open-loop damages | **CONFIRMED** |
| 7 | 81.0 % capture rejection rate / 19.0 % acceptance | **CONFIRMED** |
| 8 | per-arm yield + paired bootstrap CIs | **NEW** (no prior number to confirm) |
| 9 | Sec 4.6 "8 pages" shadow accepts | **CHANGED** → 13 accepted, 10 misclassified |
| 10 | Sec 4.6 "$7\times$ probe-score drop" | **CHANGED** → $11\times$ (0.0912 on `18.png`) |
| 11 | Sec 4.6 "previously empty outputs became fully readable" | **CHANGED** — unsupported; max gain is 1.060× |
| 12 | Table 3 "mean of two runs" | **CHANGED** — it is a single run per system; no spread exists |
| 13 | Table 3 PP-StructureV3 server RAM 8.0 vs 8.2 GB in source | **CHANGED** — flagged, not altered |
| 14 | onnxruntime 1.20.1 · i7-11800H (8C/16T) | **CONFIRMED** |
