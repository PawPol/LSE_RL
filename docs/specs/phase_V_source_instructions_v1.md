# Instructions for the Coding Agent: Rebuild the Experimental Section for a NeurIPS-Grade Submission

## Mission

Redesign the experiments so they produce a clear empirical story that is tightly aligned with the paper’s theory.

Do **not** optimize for “making the safe method win” by arbitrary benchmark search. Optimize for the following scientific objective:

> **Find tasks where the theorem predicts a nontrivial policy difference, then test whether that difference translates into planning and RL behavior.**

The current draft already shows that broad Phase I–III coverage mostly demonstrates the calibration machinery and that the deployed safe operator is near-classical on those tasks. That is useful as a sanity check, but it is not a strong main-paper empirical story. Your job is to replace the current “many tasks, mostly null outcomes” setup with a smaller number of experiments that each answer a precise question.

---

## What the current experiments already establish

Treat the following as the starting diagnosis, not as hypotheses to rediscover:

1. On the Phase I–III suite, the deployed clipped schedule is effectively in the **near-zero-β regime**, so the safe operator is numerically close to the classical Bellman operator.
2. Planning residuals and broad RL outcomes on those tasks are therefore mostly **indistinguishable** from classical baselines.
3. The Appendix B dense-cost follow-up can drive deployed `beta_tilde` away from zero, but **activation alone does not imply outcome translation**.
4. The current task search objective is misaligned: it favors activation surrogates such as mean `|u|` and informative-stage fraction, but these can increase while policy and outcome differences remain zero.
5. There are experimental-consistency issues that must be fixed before any new results are trusted.

---

## Non-negotiable rules

1. **Do not use the current 8-task x 4-algorithm matrix as the main result.** Keep it only as an appendix sanity check.
2. **Do exact planning before RL.** If a candidate task does not produce a nontrivial exact policy/value difference, do not run RL on it.
3. **Do not promote tasks based only on activation surrogates.** Promotion must depend on policy disagreement and exact value differences.
4. **Do not search until something “wins.”** Predeclare screening criteria, evaluate all tasks that pass, and report all shortlisted tasks.
5. **Do not hide tiny quantities by rounding.** Use scientific notation for small `beta_tilde`, value gaps, and return deltas.
6. **Do not bury inconsistencies.** The pipeline must fail loudly on table/figure/text mismatches.
7. **Do not spend main-text space on wall-clock and sweep counts** unless they demonstrate a real mechanism-level difference.
8. **Do not let the paper’s empirical story depend on weak seed counts.** Use paired seeds and adequate statistical power.

---

## Primary empirical story to target

The main paper should be organized around three questions:

1. **Mechanism:** When does the safe weighted-LSE operator produce a nontrivial change in continuation behavior relative to classical discounting?
2. **Translation to decisions:** When does that mechanism change the optimal policy or the exact start-state value?
3. **Safe vs. raw:** When does clipping preserve the useful nonlinear effect while preventing instability or expansiveness?

The target is a **positive mechanism-backed paper**. The experiments do **not** need to show broad benchmark superiority across many generic tasks. They **do** need to show, on a small number of theorem-linked constructed families, that the deployed clipped operator can produce:

- a nontrivial continuation change,
- exact policy/value translation under clipping,
- and a safe-vs-raw stability advantage.

If the first search pass does not yet produce such tasks, do **not** pivot to a negative-result narrative. Instead, iterate on family design, boundary tracing, and exact-planning search until you obtain a small set of promotable tasks with clear translation.

## Work package 0: Consistency audit

Before generating new experiments, audit the existing pipeline.

### Required checks

1. Recompute every table and figure directly from raw logs.
2. Fail the pipeline if any `mean_effective_discount > 1 + 1e-8` for a supposedly safe deployed operator.
3. Fail the pipeline if any stagewise realized continuation derivative exceeds its certified bound by more than tolerance.
4. Fail the pipeline if task metadata disagree across config files, tables, and captions. This includes:
   - `R_max`
   - horizon `T`
   - state/action counts
   - seed counts
   - alpha/headroom values
5. Fail the pipeline if text claims and plotted/tabled numbers disagree.
6. Emit a machine-readable and human-readable audit report.

### Deliverables

- `results/audit/consistency_report.md`
- `results/audit/consistency_report.json`
- `results/audit/recomputed_tables/`
- `results/audit/recomputed_figures/`

If the audit fails, fix the pipeline before running new searches.

---

## Work package 1: Replace the task-search objective

The current task search is too weak because it scores candidates by activation proxies that are not directly tied to the paper’s theorem-level claim.

### Old search objective to stop using as the promotion criterion

- predicted mean `|u|`
- fraction of informative stages
- any other activation-only proxy without exact policy/value translation

These can still be logged for diagnostics, but they must not determine the shortlist.

### New search objective

For each candidate finite-horizon MDP, compute **exact** classical and safe solutions on the time-augmented model and score the task using the following theorem-linked metrics.

Let `pi_cl*` be the classical optimal policy, `pi_safe*` the safe optimal policy, and let the reference occupancy be

`d_ref = 0.5 * d^{pi_cl*} + 0.5 * d^{pi_safe*}`.

Use `d_ref` to avoid evaluating only on states visited by one policy.

### Required candidate metrics

1. **Occupancy-weighted positive signed margin**
   - `margin_pos = E_{d_ref}[max(beta_tilde_t * (r - V_next_ref), 0)]`
   - Also log a normalized version divided by a task-scale constant.

2. **Realized continuation change**
   - `delta_d = E_{d_ref}[|d_safe_t - gamma|]`
   - Also log `mass_delta_d`, the occupancy mass where `|d_safe_t - gamma| > 1e-3`.

3. **Policy disagreement**
   - `policy_disagreement = P_{d_ref}(argmax_a Q_safe != argmax_a Q_classical)`
   - Also log whether the **start-state greedy action differs**.

4. **Exact value translation**
   - `value_gap = V_safe_0(s0) - V_classical_0(s0)`
   - Also log a normalized value gap.

5. **Clipping diagnostics**
   - clip fraction on visited transitions
   - fraction of states/transitions where clipping is inactive
   - fraction where clipping saturates

6. **Safety stress diagnostics**
   - raw operator local derivative statistics
   - whether raw value iteration diverges, oscillates, or converges very slowly

### Initial promotion thresholds

Use these thresholds for the first search pass and do **not** relax them post hoc after looking at final RL outcomes.

A task is promotable only if all of the following hold:

1. `policy_disagreement >= 0.05` under `d_ref` **or** the start-state greedy action differs.
2. `mass_delta_d >= 0.10`.
3. normalized `|value_gap| >= 0.005`.
4. clip fraction is in a nondegenerate band, e.g. between `0.05` and `0.80` on `d_ref` mass.
5. no certification violation occurs.

If too few tasks pass, report that outcome honestly. Do **not** silently move thresholds.

### Deliverables

- `results/search/candidate_grid.parquet`
- `results/search/candidate_metrics.parquet`
- `results/search/shortlist.csv`
- `results/search/shortlist_report.md`

---

## Work package 2: Design better candidate task families

Do **not** just shrink `R_max` and rerun the old dense-cost chains. Smaller `R_max` can enlarge the certified cap, but it can also shrink the key `(r - V)` factor and leave policy differences at zero.

Instead, construct candidate families where the classical decision boundary is intentionally close to indifferent and the nonlinear operator can plausibly flip the preferred action.

### Required candidate families

#### Family A: Aligned delayed-reward propagation

Goal: show that on aligned regions, the safe operator propagates a delayed signal backward differently from classical discounting.

Design requirements:
- sparse or delayed reward
- one or more decision points close to classical indifference
- controllable horizon and branch length
- tunable reward timing and scale
- exact model available

Vary:
- horizon `T`
- branch lengths
- reward timing
- `R_max`
- near-tie gap between competing actions
- stochastic transition probability

#### Family B: Policy-flip control tasks

Goal: produce tasks where the nonlinear Bellman target changes the optimal action at one or more reachable states.

Design requirements:
- at least one reachable state with classical action gap near zero
- nonlinearity should favor one branch through the signed-margin mechanism
- the flip should survive safe clipping

A good pattern is a forked chain or forked grid where:
- branch 1 has slightly better classical continuation,
- branch 2 has stage-local structure that the safe operator amplifies,
- the classical action gap is small enough that the nonlinear correction can matter.

#### Family C: Misaligned stress / safety tasks

Goal: show why safe clipping is needed.

Design requirements:
- a raw weighted-LSE configuration that enters an expansive or unstable regime,
- a safe clipped counterpart that remains stable,
- identical environment and raw schedule so the difference is attributable to clipping.

Measure instability explicitly rather than just showing a noisy return curve.

### Important design principle

For at least some tasks, tune the environment so that the **optimal policy itself depends on the operator nonlinearity**, not just the logged activation metrics.

---

## Work package 3: Change the planning experiments

The current “planning efficiency” story is too weak. On acyclic finite-horizon tabular problems, sweep counts and wall-clock do not show the paper’s distinctive mechanism.

### Stop using as primary planning metrics

- number of sweeps to convergence
- final Bellman residual only
- wall-clock on tiny tabular problems

These can appear in an appendix if needed.

### New primary planning metrics

For exact-planning tasks, measure the following as a function of limited backups `k`:

1. `||V_k - V*||_inf` by stage or by distance-to-terminal
2. greedy action correctness rate by stage
3. earliest stage at which the correct action is recovered
4. backward propagation distance of a rare reward/warning signal after `k` backups
5. occupancy-weighted value error under the reference occupancy
6. policy disagreement trajectory as backups proceed

### Required planning plots

1. **Propagation plot:** error versus backup depth for classical vs safe.
2. **Stagewise error heatmap:** stage on x-axis, backup count on y-axis, color = value error.
3. **Greedy-action recovery plot:** fraction of states with correct greedy action versus backup count.
4. **Safe-vs-raw stability plot:** residual/oscillation/divergence behavior for raw and clipped operators.

### Deliverables

- `results/planning/limited_backup_metrics.parquet`
- `figures/main/fig_propagation_curve.pdf`
- `figures/main/fig_stagewise_error_heatmap.pdf`
- `figures/main/fig_safe_vs_raw_stability.pdf`

---

## Work package 4: Simplify and strengthen the baselines

Do not compare against a long menu of weakly differentiated algorithms. Use a small set of baselines that answer the obvious reviewer questions.

### Required baselines

1. **Classical baseline:** `beta = 0`
2. **Safe-zero control:** same implementation path, but with zero nonlinear effect
3. **Tuned fixed-discount baseline:** classical Bellman/RL with tuned fixed `gamma_eff`
4. **Multi-step baseline:** `n`-step return or `TD(lambda)` style baseline
5. **Raw-unclipped weighted-LSE:** only on tasks where safety/stability is being tested

### Optional baseline

- A constant-small-`beta` baseline can remain as an appendix ablation, but it should not be a main comparison unless it answers a concrete question.

### Baseline philosophy

The reviewer must not be able to say:
- “This is just a smaller fixed discount.”
- “This is just a longer backup.”
- “This is just instability from a raw nonlinear operator.”

Your baseline set should preempt those objections directly.

---

## Work package 5: RL only on shortlisted tasks

Do not run RL on broad task families that exact planning has already shown to be essentially classical.

### RL gate

Only run RL on tasks that passed the exact-planning promotion thresholds.

### RL protocol

1. Use **2–3 shortlisted tasks only**.
2. Use **paired seeds / common randomness** across all arms.
3. Use the same training budget, exploration schedule, initialization, and evaluation cadence across arms.
4. Use at least **20 seeds** for final RL figures. If the pilot effect size is still tiny, go higher rather than making claims from noise.
5. Report paired-difference confidence intervals, not just per-arm mean ± std.

### RL metrics

Primary:
- paired mean return difference
- AUC return difference
- time-to-threshold difference
- instability/failure rate

Secondary:
- final return
- raw learning curves

Mechanism-linked RL diagnostics:
- distribution of realized `d_t`
- occupancy-weighted signed margin during learning
- fraction of visited transitions with `d_t < gamma`
- fraction of updates affected by clipping

### Required RL comparisons

For each shortlisted task:
- classical
- safe-zero
- safe-nonlinear
- tuned fixed-discount baseline
- multi-step baseline
- raw-unclipped only when the safety story is being tested

### Deliverables

- `results/rl/final_runs.parquet`
- `results/rl/paired_differences.parquet`
- `figures/main/fig_rl_translation.pdf`
- `figures/main/fig_rl_mechanism_diagnostics.pdf`

---

## Work package 6: Statistical reporting

### Required reporting standard

1. Use paired statistics whenever the experiment supports them.
2. Report 95% bootstrap confidence intervals for the primary paired differences.
3. Report effect sizes, not only p-values.
4. Use scientific notation for very small values.
5. Never round a meaningful quantity to `0.0000`.
6. Keep train/test selection logic explicit.

### Seed policy

- planning: deterministic exact results plus, if applicable, tie-break sensitivity analysis
- RL: at least 20 seeds for final results
- all reported main-text tasks must have the same seed count unless there is a documented reason

---

## What belongs in the main paper vs appendix

### Main paper

Keep the main empirical section small and sharp.

Recommended main-text structure:

1. **Mechanism figure:** search space plotted with policy disagreement and value gap; highlight shortlisted tasks.
2. **Propagation figure:** safe vs classical under limited backups on an aligned task.
3. **Safety figure:** raw vs clipped behavior on a stress task.
4. **RL translation figure:** only for shortlisted tasks that exact planning says are nontrivial.
5. **Compact summary table:** shortlist metrics, baselines, and main outcomes.

### Appendix

Move the following out of the main text:
- the broad Phase I–III benchmark matrix
- wall-clock and sweep-count tables on tiny tabular tasks
- large ablation grids whose outcome is “still null because beta_tilde is near zero”
- redundant algorithm-by-task panels

The Phase I–III suite should survive only as an appendix sanity check showing that the certification guardrail behaves correctly and that the operator collapses to the classical regime when the caps are tiny.

---

## Required output files and directory layout

Use this exact structure unless there is a strong reason to change it.

```text
results/
  audit/
    consistency_report.md
    consistency_report.json
    recomputed_tables/
    recomputed_figures/
  search/
    candidate_grid.parquet
    candidate_metrics.parquet
    shortlist.csv
    shortlist_report.md
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
    fig_propagation_curve.pdf
    fig_stagewise_error_heatmap.pdf
    fig_safe_vs_raw_stability.pdf
    fig_rl_translation.pdf
    fig_rl_mechanism_diagnostics.pdf
  appendix/
    fig_phase123_sanity.pdf
    fig_appendix_ablations.pdf
```

Also emit:
- `results/summaries/experiment_manifest.json` with git commit, seed list, task configs, and exact command lines.

---

## Anti-patterns to avoid

1. **Do not** use activation metrics as a substitute for policy or value translation.
2. **Do not** present near-identical learning curves across many tasks as if quantity alone creates evidence.
3. **Do not** rely on 2–5 seed results for subtle claims.
4. **Do not** compare safe and classical on tasks where exact planning already says `beta_tilde ≈ 0` and then claim “robustness.”
5. **Do not** interpret statistically insignificant differences as supporting evidence.
6. **Do not** optimize the shortlist for return after seeing RL outcomes.
7. **Do not** suppress raw-operator failures; they are necessary for the safety story when they exist.
8. **Do not** make the main claim about generic benchmark superiority. That is not what this method currently supports.

---

## Required end state

### Positive mechanism-backed paper only

The final empirical package must support a **positive but scoped** claim. The claim should be:

- there exist structured, theorem-linked finite-horizon task families where the **deployed safe clipped weighted-LSE operator** changes the optimal policy or exact start-state value relative to classical discounting,
- the change is visible in exact planning and propagation diagnostics,
- clipping preserves the useful nonlinear effect while preventing raw instability or expansiveness,
- and at least **one** shortlisted RL task shows translation beyond noise under paired seeds and adequate power.

This is sufficient for the empirical story. The paper does **not** need to claim broad benchmark superiority. It **does** need to show a clean constructive existence result with mechanism-level evidence.

### What to do if the first pass is weak

If the initial search family does not yield promotable tasks, do **not** write or propose a negative-result paper. Instead:

1. Keep the promotion thresholds fixed.
2. Expand or refine the candidate family library.
3. Preserve classical near-ties while changing temporal reward geometry, warning structure, or catastrophe structure.
4. Use exact classical and safe **decision-boundary tracing** to locate parameter regions where the safe operator changes the preferred action.
5. Choose interior parameter points with:
   - reachable contest states,
   - nontrivial policy disagreement,
   - nontrivial normalized value gap,
   - and nondegenerate clipping.
6. Only then run RL.
7. If RL gains are weaker than planning gains, keep the RL claims **supportive** and let the exact-planning and safe-vs-raw results carry the main empirical load.

### How to iterate family design constructively

When the first family definitions are not sufficient, modify the **task construction**, not the thresholds.

Use the following constructive levers:

- **Classical tie control:** solve for a parameter that makes the classical action gap near zero at a designated reachable contest state.
- **Geometry control:** introduce a second parameter that changes the stagewise shape of `r - V` without moving the classical value much.
- **Boundary targeting:** compute both the classical and safe decision boundaries, then pick points inside the region where the safe clipped operator flips the preferred action.
- **Propagation amplification:** increase delay, horizon, warning depth, reward concentration, or branch asymmetry within certified bounds.
- **Stability contrast:** include at least one raw-stress family where the raw operator becomes unstable but the clipped operator remains stable.

Suggested family expansion order if A/B/C are not enough:

1. refine **Family A** with a shape basis that preserves classical discounted value while changing reward concentration over time;
2. refine **Family B** with warning states, multi-event catastrophe branches, or delay–probability tradeoffs;
3. keep **Family C** as the safety/stability family;
4. add **Family D** for matched-classical-value but different temporal concentration / revelation structure;
5. add **Family E** for regime-shift / warning revelation tasks.

### Non-negotiable framing rule

Do **not** frame the empirical section as a negative result.

Do **not** say that activation without translation is the final paper story.

Do **not** stop after the first empty shortlist unless there is a genuine theorem–implementation mismatch or an actual resource failure.

The required deliverable is a **positive mechanism-backed empirical section** with tightly scoped claims.

## Final instruction

The single most important change is this:

> **Stop asking “Where does safe beat classical on return?”**
>
> **Start asking “Where does the theorem imply a nontrivial policy/value difference, and does that difference survive safe clipping and show up in planning or RL?”**

All experiment design, task search, metric selection, and plotting must be aligned to that question.
