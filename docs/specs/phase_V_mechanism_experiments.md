# Phase V — Mechanism-first Experiments (canonical spec)

**Status:** active. Single source of truth for the Phase V empirical rebuild.
**Supersedes (for main-paper empirical story):** Phase I, II, III, IV-A, IV-B, IV-C.
Those phases survive as **appendix sanity material only**.

**Inputs that define this spec:**
- `tasks/coding_agent_experiment_instructions.md` (initial instruction doc)
- `tasks/coding_agent_experiment_instructions_update.md` (expanded instruction doc)
- `tasks/reply_to_coding_agent_phase_v.md` (user reply fixing decisions)

If the three inputs ever conflict, the **reply** wins, then the **update**, then the initial instructions.

---

## 0. Scientific objective

> **Find tasks where the theorem predicts a nontrivial policy difference, then test whether that difference translates into planning and RL behavior.**

The main paper is organized around three questions:

1. **Mechanism.** When does the safe weighted-LSE operator produce a nontrivial change in continuation behavior relative to classical discounting?
2. **Translation to decisions.** When does that mechanism change the optimal policy or the exact start-state value?
3. **Safe vs. raw.** When does clipping preserve the useful nonlinear effect while preventing instability or expansiveness?

**The only acceptable end state is a positive, scoped, mechanism-backed paper.** Negative-result framing is not allowed. If the first search pass produces no promotable tasks, the response is to refine the task families — not to relax the promotion thresholds and not to reframe the paper as a null.

---

## 1. What Phase I–IV already established (diagnosis, not hypotheses)

1. On the Phase I–III suite, the deployed clipped schedule sits in the near-zero-β regime, so the safe operator is numerically close to the classical Bellman operator.
2. Planning residuals and broad RL outcomes on those tasks are mostly indistinguishable from classical baselines.
3. The Appendix B dense-cost follow-up (Phase IV-B) can drive deployed `β̃` away from zero, but **activation alone does not imply outcome translation**.
4. The prior task-search objective is misaligned: it scores candidates by activation proxies (mean `|u|`, informative-stage fraction) that can increase while policy and outcome differences remain zero.
5. The pipeline has consistency issues that must be fixed before any new results are trusted.

---

## 2. Non-negotiable rules

1. The current 8-task × 4-algorithm matrix is **appendix sanity only**; never a main result.
2. Do exact planning before RL. If a candidate task does not produce a nontrivial exact policy/value difference, do not run RL on it.
3. Promotion to the shortlist **must** depend on policy disagreement and exact value differences. Activation surrogates are diagnostics, not promotion criteria.
4. Do not search until something "wins." Predeclare screening criteria, evaluate all tasks that pass, and report all shortlisted tasks.
5. Do not hide tiny quantities by rounding. Use scientific notation for small `β̃`, value gaps, and return deltas. Never print `0.0000`.
6. Do not bury inconsistencies. The pipeline must fail loudly on table/figure/text mismatches.
7. Do not spend main-text space on wall-clock and sweep counts unless they demonstrate a real mechanism-level difference.
8. Do not let the paper's empirical story depend on weak seed counts. Use paired seeds and adequate statistical power (≥ 20 seeds for final RL).
9. The shortlist **must not be optimized for RL return after seeing RL outcomes**.
10. Raw-operator failures are necessary evidence for the safety story; never suppress them.

---

## 3. Core conceptual clarification: "near-indifferent classical decision boundary"

For a reachable contest state `x = (t, s)` and two candidate actions `a₁, a₂`, with task-family parameters `θ`, define the classical action gap

$$
\Delta_0(x; \theta) = Q^*_{0,\gamma}(x, a_1; \theta) - Q^*_{0,\gamma}(x, a_2; \theta).
$$

- The **classical decision boundary** is the locus of `θ` with `Δ₀(x; θ) = 0`.
- The classical planner is **close to indifferent** when `|Δ₀|` is small.
- The **near-indifference region** is a small neighborhood around the boundary — this is where a nonlinear correction has a chance to flip the ranking.

### Practical operationalization

For each candidate contest state, compute
- `gap_abs = |Δ₀|`
- `gap_norm = |Δ₀| / reward_scale`, with `reward_scale = max(R_max, |V*_{0,γ,0}(s₀)|, 1e-8)`.

### Near-tie prefilter (fixed)
- strict near-tie band: `gap_norm ≤ 0.01`
- exploratory soft band: `gap_norm ≤ 0.02` (exploration only — final promotion requires a genuine near-tie or an explicitly justified equivalent)

### Reachability (fixed)

For the designated contest state, compute its occupancy under

$$
d_\text{ref} = \tfrac12 d^{\pi^*_\text{cl}} + \tfrac12 d^{\pi^*_\text{safe}}.
$$

Require either
- contest-state occupancy under `d_ref ≥ 0.05`, **or**
- start-state greedy action differs.

---

## 4. What the search optimizes for (and what it does NOT)

### Banned as promotion criteria
- predicted mean `|u|`
- informative-stage fraction
- mean absolute `β̃`
- any activation-only proxy

These may still be **logged** for diagnostics, but they must not drive the shortlist.

### Required, theorem-linked promotion criteria
A task enters the shortlist only when **all** of the following hold (thresholds fixed):

1. `policy_disagreement ≥ 0.05` under `d_ref` **or** start-state greedy action differs.
2. `mass_delta_d ≥ 0.10`.
3. normalized `|value_gap| ≥ 0.005`.
4. designated contest state satisfies `gap_norm ≤ 0.01` (strict), or `≤ 0.02` with explicit justification.
5. designated contest state is reachable per §3, or start-state flip.
6. clip fraction on `d_ref` mass lies in `[0.05, 0.80]`.
7. no certification violation occurs.

If too few tasks pass: **report that internally, refine families, and rerun.** Do not silently move thresholds.

---

## 5. Construction recipe for candidate families

For each family, define two knobs:

- `λ` — **tie parameter.** Moves the classical action gap toward zero at a designated reachable contest state.
- `ψ` — **geometry parameter.** Changes timing / concentration / sign / stochastic shape so the nonlinear Bellman target sees the two branches differently, while holding classical value nearly fixed.

Algorithm:
1. Fix `ψ`.
2. Solve for `λ_tie(ψ)` such that `Δ₀(x_c; λ_tie(ψ), ψ) ≈ 0` — bisection over exact DP, warm-started with closed-form ties when available.
3. Sweep `λ = λ_tie(ψ) + ε` over a narrow, symmetric band of small offsets `ε`.
4. For each `(λ, ψ)`, solve exact classical optimum, exact safe optimum under the **deployed clipped schedule**, and optionally the raw optimum.
5. Compute all §6 metrics and the §4 promotion gate.
6. Promote only families where the **safe clipped operator** causes a nontrivial reachable difference.

### Required families (initial set)

#### Family A — delayed jackpot vs smooth stream (aligned propagation)
Two-branch contest at `x_c`:
- **A:** zero until terminal step, then reward `R` at depth `L`.
- **B:** per-step reward `c` for `L` steps.

Closed-form classical tie:
$$
c_\text{tie} = \gamma^{L-1} R \cdot \tfrac{1-\gamma}{1-\gamma^L}.
$$

Required extension — **shape-preserving perturbation basis**: introduce a basis `h_k(ψ)` with `Σₖ γᵏ h_k(ψ) = 0`, then set `c_k(ψ) = c_tie + h_k(ψ)`. This preserves classical discounted value while changing temporal concentration. Examples: front-loaded-with-compensating-back-subtraction, one-bump vs two-bump, smooth ramp vs flat stream.

Parameters to sweep: `L, R, γ, ε, ψ`, optional stochastic transition `p`.

#### Family B — rare catastrophe vs safe branch (policy-flip)
- **A:** immediate bonus `b`, then catastrophe `-C` at delay `L` with probability `p`.
- **B:** deterministic safe payoff `b_safe` or a short safe stream.

Closed-form classical tie:
$$
b_\text{safe} = b - p \gamma^{L-1} C.
$$

Required variants: deterministic warning state before catastrophe; shallow early reward with latent risk; multiple small catastrophes instead of a single event; matched-classical-value but different return concentration.

#### Family C — misaligned stress / safety
Designed so that a raw weighted-LSE schedule with larger temperature enters a locally expansive or unstable region on **visited** states, while the safe clipped counterpart remains stable. Environment and raw schedule are otherwise identical to isolate the effect of clipping.

Force the stress regime by engineering the visited branch so that a substantial share of visited transitions satisfy: negative or misaligned signed margin, local continuation derivative above the safe certificate, and preferably local derivative above 1 on some visited region.

### Family expansion order (if A/B/C are insufficient)
Per the reply directive, if the first pass produces too few promotable tasks:

1. Refine **Family A** (shape basis, larger `L`, sharper concentration).
2. Refine **Family B** (warning depth, multi-event branch, delay–probability trade-off).
3. Keep **Family C** as the safety/stability family.
4. Add **Family D** — matched-classical-value but different temporal concentration / revelation structure.
5. Add **Family E** — regime-shift / warning-revelation tasks.

Continue until the shortlist contains **≥ 2 promotable positive exact-planning families** and **≥ 1 safety/stability family**. Do not change thresholds.

---

## 6. Required per-candidate metrics

For each exact-planning candidate compute and persist:

| key | definition |
|---|---|
| `margin_pos` | `E_{d_ref}[ max(β̃_t · (r − V_next_ref), 0) ]` |
| `margin_pos_norm` | `margin_pos / reward_scale` |
| `delta_d` | `E_{d_ref}[ |d_safe_t − γ| ]` |
| `mass_delta_d` | `P_{d_ref}( |d_safe_t − γ| > 1e-3 )` |
| `policy_disagreement` | `P_{d_ref}( argmax_a Q_safe ≠ argmax_a Q_classical )` |
| `start_state_flip` | `1{ argmax_a Q_safe(s₀) ≠ argmax_a Q_classical(s₀) }` |
| `value_gap` | `V_safe_0(s₀) − V_classical_0(s₀)` |
| `value_gap_norm` | `value_gap / reward_scale` |
| `contest_gap_abs` | `|Δ₀(x_c)|` |
| `contest_gap_norm` | `|Δ₀(x_c)| / reward_scale` |
| `contest_occupancy_ref` | `d_ref(x_c)` |
| `clip_fraction` | fraction of visited transitions with active clip, under `d_ref` |
| `clip_inactive_fraction` | fraction where clip never binds |
| `clip_saturation_fraction` | fraction where clip saturates |
| `raw_local_deriv_stats` | mean/p50/p90/max of raw-operator local derivative on `d_ref` |
| `raw_convergence_status` | one of `{converged, oscillatory, expansive, nan_guarded, cap_reached}` |

All metrics are persisted per candidate in `results/search/candidate_metrics.parquet`.

---

## 7. Work packages

### WP0 — consistency audit (runs first)
Recompute every Phase I–IV table and figure directly from raw logs.

Fail-loud gates:
- any `mean_effective_discount > 1 + 1e-8` on a supposedly-safe deployed operator;
- any stagewise realized continuation derivative exceeds its certified bound beyond tolerance;
- metadata disagreement across configs, tables, captions (`R_max`, horizon `T`, state/action counts, seed counts, alpha, headroom);
- text claims disagreeing with plotted/tabled numbers.

Deliverables:
- `results/audit/consistency_report.md`
- `results/audit/consistency_report.json`
- `results/audit/recomputed_tables/`
- `results/audit/recomputed_figures/`
- pytest fixture `tests/audit/test_consistency_gate.py` that re-runs the gate in CI.

**If the audit fails, fix the pipeline before running Phase V search.**

### WP1 — search objective + tie solver
- `λ_tie(ψ)` bisection over exact DP, warm-started with closed-form ties for A/B.
- `d_ref` computed on the time-augmented chain and persisted as `occupancy.npz` next to every DP run.
- Metric module computing every row of the §6 table.
- `contest_state` added as a required attribute on the task-factory protocol (`experiments/weighted_lse_dp/tasks/base_families.py`).

### WP2 — task factories
- `experiments/weighted_lse_dp/tasks/family_a_jackpot_vs_stream.py`
- `experiments/weighted_lse_dp/tasks/family_b_catastrophe.py`
- `experiments/weighted_lse_dp/tasks/family_c_raw_stress.py`

Each factory exposes `contest_state`, the tie parameter, and the geometry parameter, plus a closed-form tie hint where one exists.

### WP1c — search driver + shortlist
- `experiments/weighted_lse_dp/runners/run_phase_V_search.py` replacing `run_phase4_activation_search.py` (old retained but unused).
- Cheap `contest_gap_norm` prefilter before the full DP solve.
- Hard cap: **≤ 5,000 candidate configs** aggregated across all families.
- Outputs:
  - `results/search/candidate_grid.parquet`
  - `results/search/candidate_metrics.parquet`
  - `results/search/near_indifference_catalog.csv`
  - `results/search/shortlist.csv`
  - `results/search/shortlist_report.md`
  - `results/search/phase_diagram_data.parquet`
- Empty-shortlist contract: emit `shortlist_refinement_manifest.md` listing which levers (tie / geometry / delay / concentration / warning depth / asymmetry) to refine next.

### WP3 — limited-backup planning diagnostics (main-paper planning story)
For each shortlisted task, as a function of limited backups `k`:
1. `||V_k − V*||_∞` by stage / by distance-to-terminal;
2. greedy-action correctness rate by stage;
3. earliest stage at which correct action is recovered;
4. backward propagation distance of a rare reward or warning signal after `k` backups;
5. occupancy-weighted value error under `d_ref`;
6. policy-disagreement trajectory as backups proceed.

Required plots:
- `figures/main/fig_propagation_curve.pdf`
- `figures/main/fig_stagewise_error_heatmap.pdf`
- `figures/main/fig_decision_boundary_phase_diagram.pdf`

Data:
- `results/planning/limited_backup_metrics.parquet`

### WP4 — safe-vs-raw stability
- Raw-operator VI with a hard iteration cap and NaN/overflow guard.
- Each raw run is classified into one of `{converged, oscillatory, expansive, nan_guarded, cap_reached}`.
- For Family C, compare raw vs safe-clipped on identical environments.
- Deliverables:
  - `results/planning/safe_vs_raw_stability.parquet`
  - `figures/main/fig_safe_vs_raw_stability.pdf`

### WP5 — baselines + paired RL
Required arms for every shortlisted RL task:
1. classical (`β = 0`)
2. safe-zero
3. safe-nonlinear (deployed clipped)
4. tuned fixed-discount (tuned `γ_eff`)
5. multi-step baseline (`n`-step or TD(λ))
6. raw-unclipped **only** on safety-stress tasks

Protocol:
- ≤ 3 shortlisted tasks.
- **Stage 1 pilot:** 5 seeds, paired common randomness.
- **Stage 2 final:** 20 seeds, paired common randomness. If pilot effect is tiny, go higher — do not make claims from noise.
- Same training budget, exploration schedule, initialization, and evaluation cadence across arms.
- Report paired-difference 95% bootstrap CIs and effect sizes, not per-arm mean ± std.

RL metrics:
- primary: paired mean-return difference, AUC difference, time-to-threshold difference, instability/failure rate.
- secondary: final return, raw learning curves.
- mechanism diagnostics during learning: realized `d_t` distribution, occupancy-weighted signed margin, fraction of visited transitions with `d_t < γ`, fraction of updates affected by clipping.

Deliverables:
- `results/rl/pilot_runs.parquet`
- `results/rl/final_runs.parquet`
- `results/rl/paired_differences.parquet`
- `figures/main/fig_rl_translation.pdf`
- `figures/main/fig_rl_mechanism_diagnostics.pdf`

### WP6 — paper restructure (main vs appendix)
Main text target: **5 figures + 1 table**.

Main figures:
1. `fig_mechanism_frontier.pdf` — search space with policy disagreement and value gap; shortlisted tasks highlighted.
2. `fig_decision_boundary_phase_diagram.pdf` — classical and safe action boundaries on the family parameter plane.
3. `fig_propagation_curve.pdf` — safe vs classical under limited backups on an aligned task.
4. `fig_safe_vs_raw_stability.pdf` — raw vs clipped on a stress task.
5. `fig_rl_translation.pdf` (+ companion `fig_rl_mechanism_diagnostics.pdf`) — paired RL on shortlisted tasks only.

One compact summary table: shortlist metrics, baselines, main outcomes.

Move to appendix:
- Phase I–III benchmark matrix (sanity only),
- wall-clock and sweep-count tables on tiny tabular tasks,
- large ablation grids whose outcome is "still null because β̃ ≈ 0",
- redundant algorithm-by-task panels,
- the Phase IV-B `translation_4a2` alpha sweep (archived to branch `phase-IV-alpha-sweep/archive`).

---

## 8. Directory layout (enforced)

```
results/
  audit/
    consistency_report.md
    consistency_report.json
    recomputed_tables/
    recomputed_figures/
  search/
    candidate_grid.parquet
    candidate_metrics.parquet
    near_indifference_catalog.csv
    shortlist.csv
    shortlist_report.md
    phase_diagram_data.parquet
  planning/
    limited_backup_metrics.parquet
    safe_vs_raw_stability.parquet
  rl/
    pilot_runs.parquet
    final_runs.parquet
    paired_differences.parquet
  summaries/
    experiment_manifest.json
    main_table.csv
    appendix_table_phase123.csv

figures/
  main/
    fig_mechanism_frontier.pdf
    fig_decision_boundary_phase_diagram.pdf
    fig_propagation_curve.pdf
    fig_stagewise_error_heatmap.pdf
    fig_safe_vs_raw_stability.pdf
    fig_rl_translation.pdf
    fig_rl_mechanism_diagnostics.pdf
  appendix/
    fig_phase123_sanity.pdf
    fig_appendix_ablations.pdf
```

---

## 9. Reproducibility contract

Every runner must emit `results/summaries/experiment_manifest.json` **from within the runner, not post hoc**, with at minimum:
- git SHA,
- exact argv,
- seed list,
- task config,
- calibration config,
- timestamp,
- output paths.

All experiment entry points take `--seed` and `--config`. Raw artifacts: `results/raw/<experiment>/<run_id>/` with `run.json` + `metrics.npz`. Processed aggregates: `results/processed/`. Figures: regeneratable from `scripts/` or `notebooks/`.

---

## 10. Execution order

1. WP0 consistency audit — **must PASS before anything else runs**.
2. WP1a/b metric + tie infrastructure.
3. WP2 family factories A/B/C.
4. WP1c search driver + first shortlist pass.
5. If shortlist insufficient: refine families → D/E (no threshold relaxation).
6. WP3 limited-backup planning on shortlist.
7. WP4 safe-vs-raw stability on Family C.
8. WP5 RL pilot (5 seeds) on shortlist.
9. WP5 RL final (20 seeds) on ≤ 3 tasks.
10. WP6 paper restructure + figure/table regeneration.

---

## 11. Anti-patterns (re-stated)

1. Do not use activation metrics as a substitute for policy/value translation.
2. Do not present near-identical learning curves across many tasks as if quantity alone creates evidence.
3. Do not rely on 2–5 seed results for subtle claims.
4. Do not compare safe and classical on tasks where exact planning already says `β̃ ≈ 0` and then claim "robustness."
5. Do not interpret statistically insignificant differences as supporting evidence.
6. Do not optimize the shortlist for return after seeing RL outcomes.
7. Do not suppress raw-operator failures.
8. Do not make the main claim about generic benchmark superiority.
9. Do not search only over reward scale. Search over both classical-tie **and** geometry parameters.
10. Do not claim a policy-flip is meaningful unless the flipped state is reachable.
11. Do not pivot to a negative-result paper. If the first pass is weak, refine families — the thresholds stay fixed.

---

## 12. Final instruction

> **Stop asking "Where does safe beat classical on return?"**
>
> **Start asking "Where does the theorem imply a nontrivial policy/value difference, does that difference survive safe clipping, and does it show up in planning or RL?"**

---

## 13. Planner-resolution addendum (2026-04-23)

Resolutions to ambiguities flagged by the Phase V planner:

1. **Soft near-tie band.** Only the strict band `gap_norm ≤ 0.01` promotes. The soft band `≤ 0.02` is exploration-only; promotion from the soft band requires (a) an explicit plotter-analyst note in `shortlist_report.md` justifying why the state still acts as a near-indifference pivot, and (b) orchestrator sign-off. Default: strict only.
2. **Family C necessity.** The stopping rule is **conjunctive**: the shortlist must contain ≥ 2 promotable positive exact-planning families **and** ≥ 1 safety/stability family before WP5 RL may start. If A/B exceed 2 but C is missing, the refinement loop continues on Family C until a safety task promotes or the lever space is exhausted.
3. **5,000-candidate cap.** Per-search-iteration, not lifetime. Each refinement pass resets the budget. `run_phase_V_search.py` enforces the cap per invocation and emits a warning (not a failure) if any family exceeds 60% of the cap alone.
4. **Manifest retroactivity.** The in-runner `experiment_manifest.json` contract is **forward-looking** and applies to every Phase V runner. WP0 flags every Phase I–IV runner that emits a manifest post hoc as a **MINOR drift finding** in `consistency_report.md` — to be remediated only if that Phase I–IV code is touched again, not as a blocking gate.
