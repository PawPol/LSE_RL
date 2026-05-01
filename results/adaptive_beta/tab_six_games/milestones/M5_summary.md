# Phase VIII M5 — Milestone Summary

**Status:** COMPLETE
**Date opened:** 2026-05-01T01:17:30Z
**Date closed:** 2026-05-01T01:36:00Z (approx)
**Wall-clock:** ~19 minutes
**Branch:** phase-VIII-tab-six-games-2026-04-30

## Headline

```
analysis modules added:        6 (aggregate.py + 5 plotting scaffolds)
new tests:                     30 (10 schema parity + 20 figure smoke)
test sweep:                    894 passed + 1 skipped (full repo)
M5 verdict:                    PASS (light verifier gate; no Codex review)
```

## Wave structure

| Wave | Task | Role | Model | Output | Verdict |
|---|---|---|---|---|---|
| 1 | M5 W1.A | experiment-runner | Opus 4.7 | aggregate.py 530 lines (long-CSV aggregator) + analysis/__init__.py | PASS |
| 1 | M5 W1.B | plotter-analyst | Opus 4.7 | 5 plot scripts (beta_sweep, learning_curves, contraction, sign_switching, safety_catastrophe) | PASS |
| 2 | M5 W2 | test-author | codex gpt-5.5 xhigh | test_aggregate_schema_parity.py (10) + test_figure_smoke.py (20) | PASS |

**Verifier gate:** orchestrator pytest sweep on the new test files: 30/30 PASS, 2.84s. Full repo sweep at this commit: **894 passed + 1 skipped, 10.30s**. M5 close PASS.

## Aggregator (spec §8.3)

```
analysis/aggregate.py                    530 lines
public API:
  aggregate_to_long_csv(raw_root, out_path, roster=None,
                        *, include_phase_VII=False) -> Dict
column count:                            49 (verbatim spec §8.3)
default raw_root:                        results/adaptive_beta/tab_six_games/raw (lessons.md #11)
features:
  - per-episode expansion + metadata replication
  - schema-drift detection (lessons.md #16)
  - missing-metrics-npz accounting (no silent drops, lessons.md #20)
  - gzip output support
  - Phase VII read-only narrative cross-reference (spec §10.4 / addendum §10.5)
  - CLI entry point with argparse
```

## Plot scaffolds (spec §12)

```
analysis/beta_sweep_plots.py        beta_vs_auc.pdf + beta_vs_contraction.pdf  (M6)
analysis/learning_curves.py         main_learning_curves.pdf                    (M7)
analysis/contraction_plots.py       contraction_ucb_arm_probs.pdf + ucb_lc.pdf  (M10)
analysis/sign_switching_plots.py    switch_aligned_return.pdf + beta.pdf +      (M9)
                                    beta_sign_accuracy.pdf
analysis/safety_catastrophe.py      catastrophic_episodes.pdf + clip_freq.pdf + (cross-cutting)
                                    worst_window_pctile.pdf
total panels:                       11 (all produce non-empty PDFs on the
                                        fixture; no --demo synth path,
                                        lessons.md #16)
```

## Test coverage

```
test_aggregate_schema_parity.py           10 tests (LONG_CSV_COLUMNS exact list,
                                                    default raw_root regression,
                                                    schema-drift detection,
                                                    missing-metrics-npz accounting,
                                                    gzip parity, pandas round-trip,
                                                    Phase VII read-only inclusion)
test_figure_smoke.py                      20 tests (5 modules × 4 invariants:
                                                    make_figure runs, outputs are PDFs,
                                                    PDFs non-empty, no demo synth path)
```

## Key invariants honored

- **Operator-touch alarm: clean** across all 3 dispatches.
- **Single TD-update path: clean.** No agent edits.
- **Default base_path = `results/adaptive_beta/tab_six_games`** (lessons.md #11; test_default_raw_root passes).
- **No --demo synth path** in plot scripts (lessons.md #16; test_figure_smoke verifies).
- **Schema parity validated** byte-identically against spec §8.3 column list.
- **All pytest under `.venv/bin/python`** (lessons.md #1).

## Bug-hunt protocol disposition

No suspicious-result triggers (T1–T10) fire on M5 — analysis-scaffold milestone with no experimental results yet. M6 is the first stage where T1–T10 can fire.

## Token + dispatch budget consumed (M5)

| Role | Model | Dispatches | Tokens (est.) |
|---|---|---:|---:|
| experiment-runner | Opus 4.7 | 1 (M5.W1.A) | ~88,000 |
| plotter-analyst | Opus 4.7 | 1 (M5.W1.B) | ~65,000 |
| test-author | codex gpt-5.5 xhigh | 1 (M5.W2) | ~85,000 |
| **M5 total** | | **3** | **~238,000** |

**Cumulative (M0–M5):** 33/300 dispatches; ~2.7 M tokens; ~3.5 hours of 36h cap.

## Next: M6 (compute-heavy — Stage 1 β-grid sweep)

Per addendum overnight mode, M5 close is silent. Auto-proceed to M6.

M6 dispatch plan (per spec §M6):
1. **Wave 1:** experiment-runner builds the M6 runner script + configs:
   - `experiments/adaptive_beta/tab_six_games/runners/run_phase_VIII_stage1_beta_sweep.py` (NEW)
   - `experiments/adaptive_beta/tab_six_games/configs/dev.yaml` (Stage A dev: 3 seeds × 1k eps)
   - `experiments/adaptive_beta/tab_six_games/configs/stage1_beta_sweep.yaml` (Stage 1 main: 10 seeds × 10k eps)
2. **Wave 2 (Stage A dev pass):** 6 games × 3 subcases × 7 arms × 3 seeds × 1k eps = 378 runs.
   Estimated wall-clock: ~5-15 min on parallel CPU.
3. **Wave 3 (auto-promote check per addendum §4.1):** P1-P6 sanity criteria. If green → Stage 1 main. If fail → halt.
4. **Wave 4 (Stage 1 main pass):** 6 games × 3 subcases × 7 arms × 10 seeds × 10k eps = 1,260 runs.
   Estimated wall-clock: ~2-4 hours on parallel CPU.
5. **Wave 5 (bug-hunt T1-T10):** orchestrator runs the suspicious-result detector after Stage 1 main results land. Triggers fire focused Codex bug-hunt reviews per addendum §3.1.
6. **Wave 6 (Codex review G6c, conditional):** if results flagged for main paper.
7. **Wave 7 (figures + tables):** plotter-analyst runs the M6 plot scripts on the aggregated long-CSV.
8. **M6 close commit + push.**

M6 is the compute-bound milestone. Wall-clock estimate per addendum §5.1: 4-8 hours total.
