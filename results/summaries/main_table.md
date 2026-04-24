# Phase V + VI Main-Paper Summary Table

One row per family grouping with headline metrics. `promotion_mode`:
`binding_clip` (clipping active in [5%, 80%]) or `safe_active_no_distortion` (raw already within cert; no clip).

| family | interpretation | n_tasks | promotion_mode | max \|vgn\| | max pol_disagree | flip rate | max d_raw | max clip frac | mean gap_cl_eval | mean gap_safe_eval |
|---|---|---|---|---|---|---|---|---|---|---|
| A (det.) | Jackpot vs smooth stream, deterministic | 5 | safe_active_no_distortion | 0.0078 | 0.2500 | 1.0000 | 0.9544 | 0.0000 | — | — |
| A (stoch., VI-E) | Jackpot vs smooth stream, p_transit<1 | 6 | safe_active_no_distortion | 0.0062 | 0.2500 | 1.0000 | 0.9525 | 0.0000 | 2.96e-16 | 0.1560 |
| C (safety, VI-A) | Raw-stress: safe preserves cert, raw breaches it | 84 | binding_clip | — | — | — | 1.2848 | 1.0000 | — | — |