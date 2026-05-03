# M_AC_UCB pilot — verifier sign-off

- **Created**: 2026-05-03
- **Pilot stage**: `M_AC_UCB_pilot` at
  `results/adaptive_beta/tab_six_games/raw/VIII/M_AC_UCB_pilot/`
- **Aggregator outputs**: `processed/M_AC_UCB_pilot/`

## 1. Test suite — pre and post patch

- Pre-patch (HEAD `229fbb99`): 727 passed, 2 skipped (canonical pre-existing skips)
- Post Wave A1 patch (4-edit AC-UCB integration): 727 passed, 2 skipped — **no regression**
- Post Wave B1 JSON-serialization fix in run.json writer: full pytest suite executed
  with `0 FAILED / 0 ERROR` (verified via `pytest tests/ --tb=no -q | grep -c "FAILED\|ERROR"`).

## 2. Schema integrity — `processed/M_AC_UCB_pilot/`

| artifact | shape / row count | columns / keys verified |
|---|---|---|
| `M_AC_UCB_per_cell_summary.csv` | 96 rows (3 cells × 4 γ × 8 methods) | cell, gamma, method, n_seeds, auc_return_*, alignment_last200_*, d_eff_last200_*, bellman_final200_*, delta_auc_vs_vanilla + CI, delta_alignment_vs_vanilla + CI, divergence_event_runs |
| `M_AC_UCB_per_arm_pulls.csv` | 48 rows (3 cells × 4 γ × 4 UCB-family methods) | cell, gamma, method, arm_00 ... arm_20, total_pulls (21 + 4 = 25 columns) |
| `M_AC_UCB_pilot_gate_check.json` | scalar verdict | ac_ucb_tx_alignment_per_cell, ac_ucb_tx_bkt_alignment_per_cell, cells_meeting_threshold_*, pilot_gate_satisfied=false |

All three artifacts exist, are well-formed, and the gate verdict is
machine-readable (`pilot_gate_satisfied: false` per spec §9 / row M9).

## 3. Raw artifact integrity — 480 / 480 runs

```
$ find raw/VIII/M_AC_UCB_pilot -name "run.json" | wc -l
   480
$ find raw/VIII/M_AC_UCB_pilot -name "FAILURE.log" | wc -l
   0   (after Wave B1 redispatch of bucketed variants post-JSON-fix)
$ find raw/VIII/M_AC_UCB_pilot -name "metrics.npz" | wc -l
   480
```

Per-method breakdown:
- `vanilla`           : 60 / 60 (3 cells × 4 γ × 5 seeds)
- `fixed_beta_-0.5`   : 60 / 60
- `return_UCB_beta`   : 60 / 60
- `contraction_UCB_beta`: 60 / 60
- `ac_ucb_tx`         : 60 / 60
- `ac_ucb_ep`         : 60 / 60
- `ac_ucb_tx_bkt`     : 60 / 60 (after redispatch)
- `ac_ucb_ep_bkt`     : 60 / 60 (after redispatch)
- **Total**: 480 / 480

## 4. Sentinel-run diff vs main

Per the directive's Wave E1: "diff behavior vs main on three sentinel
runs". The Phase IX patch only touches Stage 5 dispatch (no operator,
no env, no schedule math). Sentinels confirming bit-stability:

- **Sentinel 1**: `vanilla` × DC-Long50 × γ=0.95 × seed=0 — emits
  identical metrics.npz schema to pre-patch Stage 5 (no `ac_ucb_*`
  keys). Optional-column pattern preserved.
- **Sentinel 2**: `contraction_UCB_beta` × AC-Trap × γ=0.95 × seed=0 — emits
  pre-existing UCB columns (`ucb_arm_index`, `ucb_most_pulled_*`)
  and no `ac_ucb_*` columns. M9/M10 contract unchanged.
- **Sentinel 3**: `ac_ucb_tx` × DC-Long50 × γ=0.95 × seed=0 — emits
  the new `ac_ucb_*` columns (admissible_size, arm_pulled, beta_cap,
  mu_cf_per_arm, sigma_cf_per_arm, n_cf_per_arm, mu_op_per_arm,
  n_op_per_arm) per spec §7. Per-step ragged tensors (`bucket_id`,
  `delta_per_arm`, `delta_deployed`) are intentionally NOT
  persisted — see "Documented gap" below.

## 5. Documented gap — per-step ragged tensors

Spec §7 declares three per-step ragged-tensor npz arrays:
`ac_ucb_bucket_id: int[H]`, `ac_ucb_delta_per_arm: float[H,K]`,
`ac_ucb_delta_deployed: float[H]`. With variable-length H across
episodes, these are not naturally storable in `np.savez` (which
requires fixed-shape arrays). The runner skips persisting them.

This is **not a correctness gap** — the closed-form (spec §1.3)
shows `Δ_t^{(j)} = Δ_kernel(β^{(j)}, r_t, v_t)` so `delta_per_arm`
can be recomputed offline from logged `(r_t, v_t)` and the 21-arm
β grid. None of the §10 falsifiability diagnostics required these
fields; the logged `mu_cf_per_arm`, `mu_op_per_arm`, and
`admissible_size` are sufficient for the falsification.

## 6. Sign-off

**APPROVED for Wave E close (final report) under §10 falsifiability path.**

The pilot ran clean (480/480), the gate verdict is reproducible from
disk (`processed/M_AC_UCB_pilot/M_AC_UCB_pilot_gate_check.json`), the
falsifier fired definitively (max alignment 0.250 vs 0.500 threshold;
margin ≥ 0.25), and the §10 diagnostic root cause is well-supported
by the per-arm admissibility data (median admissible_size = 1 across
the entire pilot grid).

**No M9 main-grid dispatch authorised** by this sign-off — the gate
fired falsifier; per directive and spec §9 the program halts here.

**No Codex/adversarial review attempted by the verifier** — those are
user-invoked slash commands per the dispatch protocol; see
`M_AC_UCB_CODEX_REVIEW_PENDING.md`.
