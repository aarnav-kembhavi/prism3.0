# _quarantine — MANIFEST

Files moved out of the repo root because nothing references them, or because
something else supersedes them.

**Nothing here has been deleted.** This directory is a holding area for review.
If any of it turns out to matter, move it back — the original path is recorded
for every entry.

Moved 2026-08-24 on branch `repo-handover`. Evidence originally gathered in
`RESTRUCTURE_PLAN.md` §6 and re-verified at move time.

---

## Contents

| File here | Original path | Why |
|---|---|---|
| `skill.md` | `skill.md` | Not a PRISM file at all |
| `paperresults.md` | `paperresults.md` | Superseded by `docs/paperresults.md` |
| `SESSION_HANDOFF.md` | `SESSION_HANDOFF.md` | Handoff for a branch that no longer exists |
| `vscode/settings.json` | `.vscode/settings.json` | Rule for a file that does not exist |
| `S2L (23).pdf` | `S2L (23).pdf` | Byte-identical duplicate of `figures/S2L (23).pdf` |

---

## Evidence

### `skill.md` → was `skill.md`

An Anthropic agent-skill definition, not project content. Its frontmatter reads:

```
name: frontend-design
description: Guidance for distinctive, intentional visual design when building
             new UI or reshaping an existing one.
```

The body is design guidance about typography, palettes and layout. It mentions
nothing in this repository. It was tracked in git, so it was committed by
accident rather than dropped in.

**Verified:** `git grep -F skill.md` across all tracked files returns no
inbound reference.

---

### `paperresults.md` → was `paperresults.md`

Superseded by `docs/paperresults.md`. Two different documents, not copies:

| | root (this file) | `docs/paperresults.md` |
|---|---|---|
| Title | "Paper Results: SOTA Comparison on OmniDocBench **v1.5**" | "Results vs. State of the Art" |
| Basis | "11 systems self-run", overnight run | **v19 run (2026-07-07)**, full official dataset |
| Length | 134 lines | 206 lines |

`README.md` links only `docs/paperresults.md`. Every other tracked reference —
in `ABLATION_PROVENANCE.md`, `paper.md`, `docs/context.md` — cites
`docs/paperresults.md` with a line number. Nothing cites the root copy.

**Kept because:** it is the only surviving record of the 11-system v1.5
comparison. The newer document does not contain those numbers.

---

### `SESSION_HANDOFF.md` → was `SESSION_HANDOFF.md`

Context handoff written 2026-07-07 for branch **`wacv-results-hardening`**.
That branch is gone; work has since moved through `latency-hardening` to
`repo-handover`. Its first line states the branch explicitly, so the staleness
is self-evident rather than inferred.

**Only inbound reference** is `RELEASE_REPORT.md:70`, which names it in prose as
a document *excluded* from the supplementary release — not a link, and it does
not break.

---

### `vscode/settings.json` → was `.vscode/settings.json`

The file's entire content is one terminal auto-approve rule:

```json
"/^\\.venv/Scripts/python test_rapid_ocr\\.py$/": { "approve": true }
```

It matches `.venv/Scripts/python test_rapid_ocr.py`. **Neither `.venv/` nor
`test_rapid_ocr.py` exists in this repository.** The rule can never fire.

Note it was *tracked* even though `.gitignore` lists `.vscode/` — it was
committed before that ignore rule was added, and ignore rules do not apply to
already-tracked files.

`.vscode/` is now an empty directory, left in place.

---

### `S2L (23).pdf` → was `S2L (23).pdf` (repo root)

Byte-identical to `figures/S2L (23).pdf`:

```
c47723bda0e6f3ff1767397cdd01d021  S2L (23).pdf            (2,800,798 bytes)
c47723bda0e6f3ff1767397cdd01d021  figures/S2L (23).pdf    (2,800,798 bytes)
```

`figures/` is a frozen conference-submission artifact, so the copy under
`figures/` is authoritative and was **not touched**. Only the root duplicate
moved here.

**This file is deliberately left untracked.** It was untracked at the root and
stays untracked here — committing a 2.8 MB duplicate of a frozen artifact would
add it to git history permanently for no benefit.

---

## Not quarantined, but worth knowing

These were found during the same sweep. They are **not** in this directory.

| Path | Finding | Why it was left |
|---|---|---|
| `.cache/huggingface/` | Unreferenced. No code reads `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, or a root `.cache`. Its only content is download metadata for `models/MFD/YOLO/yolo_v8_ft.pt` — the dead MFD model from §6.2 | Gitignored, 2 files, zero bytes of payload. Absent from a fresh clone, so it is not clone clutter |
| `__pycache__/app.cpython-312.pyc` | Regenerable bytecode | Gitignored, recreated on next run |
| `_web_uploads/fb2a1a08….png` | Orphaned upload left by a crashed `app.py` job | The **directory** is live — `app.py:25` sets `UPLOAD_DIR` to it and recreates it. Only the stray file is dead |
| `paper_overleaf.zip` | Stale snapshot of `paper_overleaf/` — 18 of 19 entries byte-identical, `sec/4_experiments.tex` drifted | Frozen submission artifact |
| `uv.lock` | Stale against the current `pyproject.toml` | Still the lockfile; regenerate with `uv lock` (see `SETUP.md` §1) |
| `paper.md` | Referenced by `README.md` at lines 18 and 31 | Live |

All of the first three are gitignored and never reach a teammate's clone, so
moving them would tidy only this one machine while adding noise here.
