# Codex Standard Review — Phase III R2

Session ID: 019da19f-eb97-73c1-a714-77c6f0404ae3
Base: 4fdbf0d (Phase II close commit)
Branch: phase-III/closing
Round: R2 (post R1-fix commit 11f2189)
Status: completed

## Summary

The Phase III aggregation path is currently unusable because it calls its write helpers incorrectly and invokes `aggregate_safe_stats` with an incompatible API. There is also a correctness bug in safe policy iteration when tolerance-based stopping is used, so the patch is not ready as-is.

## Findings

- [P1] Pass aggregation write helpers the arguments they expect — experiments/weighted_lse_dp/runners/aggregate_phase3.py:281-287
  `aggregate_group()` currently calls `save_json` and `save_npz_with_schema` with the argument order reversed. `save_json(summary, out_dir / "summary.json")` tries to treat the summary dict as a filesystem path, and both `save_npz_with_schema(...)` calls are also missing the required schema argument entirely, so any non-dry-run Phase III aggregation will raise before it can write outputs.

- [P1] Stop calling `aggregate_safe_stats` with the wrong signature — experiments/weighted_lse_dp/runners/aggregate_phase3.py:461-463
  The helper imported from `common.schemas` expects a safe-payload dict plus `(T, gamma)`, but here it is invoked as if it were a quantile reducer over `(stages, values, n_stages, quantiles=...)`. That raises `TypeError` as soon as a group has safe transition data, so the new Phase III aggregator cannot produce `safe_stagewise.npz` for RL runs.

- [P2] Keep `pi` consistent with `Q`/`V` when PI stops on tolerance — mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_policy_iteration.py:420-427
  When `tol > 0` and the residual threshold is hit before policy stability, `run()` assigns `self.pi = pi_new` and then breaks without re-evaluating that improved policy. The returned `Q` and `V` still correspond to the previous policy evaluation (`Q_pi`, `V_pi`), so callers get an internally inconsistent result whenever residual-based early stopping is enabled.
