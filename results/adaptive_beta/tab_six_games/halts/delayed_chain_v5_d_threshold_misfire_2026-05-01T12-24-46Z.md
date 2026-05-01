# HALT 4 — DC-Long50 v5 smoke: ordering correct, Cohen's d guard misfires

- **Halt UTC**: 2026-05-01T12:24:46Z
- **Milestone**: M2 reopen close (v5 fold-in pending)
- **Trigger class**: T11 (spec §5.7 v5) — smoke test FAILED, but on the
  effect-size guard, not on the directional prediction
- **Severity**: paper-critical / BLOCKER per addendum §6 + v5 directive
  ("If the smoke test STILL fails, that's a fourth-halt signal — HALT
  and STOP. ... Do NOT auto-patch beyond v5.")
- **HEAD at halt**: `99a98340dfe8bfc24f9f97e353b5b15b3a16fa44`
- **Working tree**: dirty — v3+v4+v5 corrections preserved uncommitted
  (atomic-commit pending researcher decision):
  - `experiments/adaptive_beta/tab_six_games/metrics.py` — v3 helpers
    (`q_convergence_rate`, `q_star_delayed_chain`) + v5 helpers
    (`bellman_residual_beta`, `auc_neg_log_residual`); `__all__`
    exports updated
  - `tests/adaptive_beta/tab_six_games/test_phase_VIII_metrics.py` —
    +4 v5 tests (`test_bellman_residual_beta_zero_at_fixed_point`,
    `test_bellman_residual_beta_classical_collapse`,
    `test_auc_neg_log_residual_monotone`,
    `test_bellman_residual_beta_handles_divergent_Q`); +
    `_make_chain_transition` helper
  - `tests/adaptive_beta/strategic_games/test_delayed_chain_smoke_prediction.py`
    — full v5 rewrite (uses `bellman_residual_beta` +
    `auc_neg_log_residual`)
  - `docs/specs/phase_VIII_tab_six_games.md` §5.7 — v3+v4 wording (NOT
    yet v5 — A.3/A.4/A.7 deferred until researcher resolves the
    threshold question)

## Empirical result (deterministic; identical across all 3 seeds)

| schedule    | β   | AUC(-log R)  | final R    |
| ----------- | --- | -----------: | ---------: |
| FixedNeg    | -1  |  +8660.5433  | 3.45e-11   |
| ZeroBeta    |  0  |  +4606.8534  | 2.45e-09   |
| FixedPos    | +1  | -23598.9510  | 2.69e+06   |

Per-seed AUCs are **bit-identical** for all three methods (printed
output: `[8660.5433, 8660.5433, 8660.5433]` etc). Intra-method std is
exactly zero.

## Directional prediction (v5)

> AUC(-log R_{β=-1}) > AUC(-log R_{β=0}) > AUC(-log R_{β=+1})

**PASS.** Both gaps strictly positive, large in absolute terms:

- gap(-1, 0)  =  8660.54  -  4606.85  =  +4053.69
- gap( 0,+1)  =  4606.85  - (-23598.95) = +28205.80
- residual final values are over 17 orders of magnitude apart between
  β=-1 (3.45e-11, fully contracted) and β=+1 (2.69e+06, divergent)

The science of the v5 prediction is empirically vindicated.

## What failed: the Cohen's d ≥ 0.3 guard on `minus` vs `zero`

Computed `pooled_std = std(concat([am, az, ap])) = 14347.57`.

- `d(minus-zero) = 4053.69 / 14347.57  =  0.2825`  ← **fails** (< 0.3)
- `d(zero-plus)  = 28205.80 / 14347.57 =  1.9659`  ← passes

## Root cause: Cohen's d is ill-defined under perfect determinism

The DC-Long50 advance-only chain has **zero stochasticity** in this
configuration:

- ``Discrete(1)`` action space — policy is forced (no policy
  randomness)
- ``ε = 0`` ε-greedy (no exploration randomness)
- deterministic chain dynamics (no transition randomness)
- ``PassiveOpponent(n_actions=1)`` — adversary is deterministic

Therefore intra-method variance is **exactly 0** for every β. The
"pooled std" that enters Cohen's d is computed over the concatenation
of all three methods' AUCs — i.e. it measures the *inter-method*
spread, which is dominated by β=+1's divergence (-23599 dragging the
spread up to ~14347).

Mechanical consequence: with zero intra-method noise, Cohen's d
collapses to `(gap_small) / (c · spread_dominated_by_largest_gap)`,
which can **never reach 0.3** if the smaller gap is less than ~30% of
the larger gap. Even an arbitrarily clean signal cannot satisfy the
guard once the "noise floor" is replaced by the divergence-amplified
mean of β=+1.

This is a guard-design mismatch with the test bed, not a failure of
the v5 metric or prediction. The v5 metric (β-specific Bellman
residual `||T_β Q − Q||_∞`) and the v5 prediction (AUC(-log R)
ordering) both behave exactly as the alignment-condition theory
predicts:

- β = -1, optimistic init Q_0=20, r=0, v≈20 → β·(r-v) = +20 > 0
  (aligned) → contraction tightens → R → 0 fast (final 3.5e-11)
- β = 0 (classical) → R → 0 with standard rate (final 2.5e-9)
- β = +1 → β·(r-v) = -20 < 0 (anti-aligned) → optimistic bootstrap
  amplified → R diverges (final 2.7e+06)

## Remediation options (researcher decision required)

The user's v5 directive is explicit: "Do NOT auto-patch beyond v5".
Hence I halt and present the choice. The options are ordered by my
recommendation strength.

### (η) Replace Cohen's d guard with a noise-free significance test

Drop `Cohen's d > 0.3` and replace with: assert that
`gap_small >= MIN_RELATIVE_GAP * gap_large` for some
`MIN_RELATIVE_GAP ∈ {0.10, 0.15}` (the empirical ratio is
4054/28206 = 0.144), or simply assert each gap exceeds an absolute
floor (e.g. `gap_small > 1000.0`). Cohen's d was meant to certify a
"meaningful" effect size in noisy regimes; in a perfectly
deterministic Discrete(1) testbed the directional assertion alone
plus an absolute-gap floor is the correct instrument.

**Pro**: preserves the v5 metric & prediction; trivial code change;
documents the deterministic-testbed caveat in spec §5.7. Empirical
results are 17 orders of magnitude apart on final R — there is no
defensible reading under which this signal is "too weak".

**Con**: requires a small spec edit (§5.7 + §23 changelog); arguably
expands beyond "v5 only" — but the change is to the *guard*, not the
metric or prediction.

### (θ) Drop only the Cohen's d guard, keep mean-ordering assertion

Remove the d-block entirely; rely on mean-ordering assertions
(`auc_minus.mean() > auc_zero.mean() > auc_plus.mean()`) as the
T11 trigger. Document in spec §5.7 that DC-Long50 is deterministic
and the mean-ordering assertion is the falsifiability criterion.

**Pro**: even smaller change than (η); honest about what determinism
buys us.
**Con**: leaves no quantitative robustness check, even an absolute
one. (η) is the better engineering choice.

### (ζ) Drop delayed_chain from M6 entirely (user's prescribed fallback)

Per user's v5 directive: "fall back to option (ζ): drop delayed_chain
from M6 entirely". This abandons the long-horizon credit-assignment
cell.

**Pro**: strictest reading of the user's halt directive.
**Con**: loses paper-headline cell ("Selective Temporal Credit
Assignment" with no long-horizon evidence). The empirical result
*supports* the paper claim — dropping the cell because of a guard
mismatch would be premature.

## Orchestrator recommendation

**(η)** — replace Cohen's d threshold with an absolute-gap or
relative-gap test that is well-defined in deterministic settings.
The v5 metric and prediction are both correct (verified by 17
orders-of-magnitude residual separation between β=-1 and β=+1, and
strictly positive directional gaps in both directions); only the
robustness guard is misaligned with the test bed.

**Decision required from user** before I:
1. land the v5 corrections atomically (commit
   `phase-VIII(M2 reopen): resolve HALT 3 — β-specific Bellman
   residual metric (v5 corrections to v3 patch)` plus the chosen
   guard variant),
2. update spec §5.7 P-Contract + T11 + §23 changelog,
3. resume autonomous M4 reopen → M6 → M7 → M12.

## Appendix: raw numerics (DC-Long50, γ=0.95, 1000 episodes, 3 seeds)

```
AUC by method (one row per seed):
  minus: ['8660.5433', '8660.5433', '8660.5433'] mean=8660.5433
  zero : ['4606.8534', '4606.8534', '4606.8534'] mean=4606.8534
  plus : ['-23598.9510', '-23598.9510', '-23598.9510'] mean=-23598.9510
final R (one row per seed):
  minus: [3.4504e-11, 3.4504e-11, 3.4504e-11]
  zero : [2.4491e-09, 2.4491e-09, 2.4491e-09]
  plus : [2.6859e+06, 2.6859e+06, 2.6859e+06]
pooled_std    = 14347.5696
d(minus-zero) = 0.2825   <-- FAILS the 0.3 guard
d(zero-plus)  = 1.9659
```
