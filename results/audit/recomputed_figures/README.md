# Recomputed Phase I-IV figures (WP0 audit)

This directory is intentionally left empty for the default WP0 run.

Per spec §7 WP0 ("recomputed figures ... optional; skip if expensive,
but log the numeric comparison"), the consistency audit does not
re-render PDFs from raw `.npz` logs. Instead, the numeric quantities
that feed every Phase I-IV figure are reproduced in the sibling
`recomputed_tables/` directory and diffed against the committed
tables. Any mismatch is logged in `consistency_report.json` under
`check == "table_diff"`.

If a future audit iteration needs pixel-level figure diffs, extend
`scripts/audit/run_consistency_audit.py` to invoke the
`experiments/weighted_lse_dp/analysis/make_phase*_figures.py`
scripts and write outputs into this directory before diffing.
