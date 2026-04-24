# Instructions for the Coding Agent: Rebuild the Experimental Section for a NeurIPS-Grade Submission (Expanded)

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


---

## Core conceptual clarification: what “classical decision boundary close to indifferent” means

This phrase has to be made precise because it is central to the new search strategy.

### Definitions

Pick a **reachable contest state** `x = (t, s)` and two candidate actions `a1, a2`. Let the task family have parameters `θ`.

Define the **classical action gap**
$$
\Delta_0(x; \theta) = Q^*_{0,\gamma}(x, a_1; \theta) - Q^*_{0,\gamma}(x, a_2; \theta).
$$

Then:

- The **classical decision boundary** is the set of parameter values `θ` such that  
  $$
  \Delta_0(x; \theta) = 0.
  $$
  That is the set where the classical planner switches which action it prefers.

- The classical planner is **close to indifferent** at `x` when `|Δ0(x; θ)|` is small.  
  In words: under the classical Bellman operator, the top two actions are almost tied.

- The **near-indifference region** is a small neighborhood around the classical decision boundary.  
  That is where a nonlinear correction has a chance to flip the ranking.

### Why this matters

If the classical action gap is uniformly bounded away from zero, then small deviations from the classical Bellman operator will not change the optimal policy. Therefore, searching in regimes where the classical policy already has a large margin is a poor way to reveal operator-driven policy changes.

This is exactly why “high activation” is not enough. You can have large `|u|`, large local signed margins, or visibly nonzero `beta_tilde` and still get **no decision change** if the classical action gap is too large.

### Practical operationalization

For each candidate contest state, compute:

- `gap_abs = |Δ0|`
- `gap_norm = |Δ0| / reward_scale`

Use a task-scale normalization such as
- `reward_scale = max(R_max, |V^*_{0,\gamma,0}(s0)|, 1e-8)`

Initial screening rule for a **near-tie** candidate:

- strict near-tie band: `gap_norm <= 0.01`
- exploratory soft band: `gap_norm <= 0.02`

The soft band can be used for exploration, but a task should not be promoted to the final shortlist unless a designated reachable contest state lies in the strict or near-strict band.

### Reachability requirement

A policy flip at a state that is never visited is not interesting. For every designated contest state, also compute its occupancy under

$$
d_{\text{ref}} = \frac{1}{2} d^{\pi^*_{\text{cl}}} + \frac{1}{2} d^{\pi^*_{\text{safe}}}.
$$

Require either:

- contest-state occupancy under `d_ref` at least `0.05`, or
- start-state greedy action differs.

---

## What the search must optimize for instead of raw activation

The new search must target **decision-sensitive operator effects**.

### Bad search target

Do **not** optimize directly for:
- predicted mean `|u|`
- informative-stage fraction
- mean absolute `beta_tilde`
- any proxy that does not test whether the operator changes a reachable decision

Those are diagnostics, not promotion criteria.

### Good search target

Search for tasks where all of the following are plausible:

1. the classical planner is close to indifferent at a reachable contest state,
2. the branch geometry makes the nonlinear operator respond differently across actions,
3. the safe clipped operator actually changes the exact policy or exact value,
4. the effect survives clipping rather than existing only in the raw operator.

---

## How to construct candidate families that can reveal policy flips

The right design pattern is:

1. **Choose a small exact-planning template** with one designated contest state.
2. **Introduce one parameter that tunes the classical tie.**
3. **Introduce a second parameter that changes the operator-sensitive branch geometry** without moving the classical value too much.
4. **Solve for the near-tie parameter exactly** under the classical planner.
5. **Sweep a narrow band around that tie** and evaluate both classical and safe exact solutions.
6. **Promote only families that show policy disagreement or nontrivial value translation after safe clipping.**

### General two-knob recipe

For each family, define:

- `λ`: **tie parameter**  
  This parameter exists only to move the classical action gap toward zero.

- `ψ`: **geometry parameter**  
  This parameter changes the timing, concentration, sign pattern, or stochastic shape of rewards so that the nonlinear Bellman target sees the two branches differently, even when classical value is matched.

Algorithm:

1. Fix `ψ`.
2. Solve for `λ_tie(ψ)` such that the designated contest state satisfies  
   $$
   \Delta_0(x_c; \lambda_{\text{tie}}(\psi), \psi) \approx 0.
   $$
   Use exact DP and either bisection or a fine grid with local interpolation.
3. Sweep `λ = λ_tie(ψ) + ε` over a narrow window of small offsets `ε`.
4. For each `(λ, ψ)`, solve:
   - exact classical optimum,
   - exact safe optimum with the **deployed clipped schedule**,
   - optionally the raw optimum for safety-stress analysis.
5. Compute:
   - classical gap,
   - policy disagreement,
   - start-state action flip indicator,
   - value gap,
   - realized continuation change,
   - clip diagnostics.
6. Keep only candidates where the safe operator causes a nontrivial reachable difference.

### Why the second knob matters

A family needs more than a tie parameter. If both branches are identical up to a small additive offset, then moving close to the classical tie just produces two nearly identical branches, which often still yields no interesting nonlinear effect.

The second knob should change **branch shape**, for example:

- reward concentration: one late jackpot versus a smooth stream
- risk shape: rare catastrophe versus safe deterministic payoff
- timing pattern: front-loaded versus back-loaded rewards
- transition structure: deterministic versus stochastic progression
- sign pattern: same classical mean, different local `(r - V)` profiles

The point is to hold classical value nearly fixed while changing the local geometry that the nonlinear Bellman target responds to.

---

## Concrete candidate family A: delayed jackpot vs smooth stream

### Template

At the contest state `x_c`, action `A` and action `B` lead to two branches of equal horizon length `L`.

- **Action A:** a delayed jackpot  
  Rewards are zero until the end, then terminal reward `R`.

- **Action B:** a smooth stream  
  Rewards are a per-step amount `c` for `L` steps.

Classical values at the contest state are:

$$
Q^0(A) = \gamma^{L-1} R
$$

and

$$
Q^0(B) = c \sum_{k=0}^{L-1} \gamma^k
       = c \frac{1-\gamma^L}{1-\gamma}.
$$

To make the classical planner exactly indifferent, choose

$$
c_{\text{tie}} = \gamma^{L-1} R \cdot \frac{1-\gamma}{1-\gamma^L}.
$$

Then sweep

$$
c = c_{\text{tie}} + \varepsilon
$$

for small positive and negative `ε`.

### Why this family is useful

The two branches can be classically tied while having very different stagewise structure:

- the jackpot branch has concentrated delayed reward,
- the stream branch has diffuse reward timing.

That changes the stagewise `r - V` geometry and therefore changes how the nonlinear Bellman target can react.

### Stronger variant: shape-preserving perturbations

Do not stop at constant streams. Introduce a shape basis `h_k(ψ)` such that

$$
\sum_{k=0}^{L-1} \gamma^k h_k(\psi) = 0.
$$

Then define

$$
c_k(\psi) = c_{\text{tie}} + h_k(\psi).
$$

This preserves the **classical discounted value** while changing the **temporal concentration** of reward. That is a better way to separate classical equivalence from operator-sensitive geometry.

Examples:

- front-loaded perturbation with compensating back-loaded subtraction
- two-bump versus one-bump reward profiles
- smooth ramp-up versus flat stream

### Parameters to sweep

- `L`
- `R`
- `γ`
- `ε`
- shape parameter `ψ`
- transition stochasticity if the chain is made probabilistic

### Required outputs for this family

- classical tie curve
- safe tie curve
- phase diagram showing where the classical and safe preferred actions differ
- limited-backup propagation plots

---

## Concrete candidate family B: rare catastrophe vs steady safe branch

### Template

At the contest state `x_c`:

- **Action A:** immediate bonus `b`, followed by a catastrophe `-C` at delay `L` with probability `p`
- **Action B:** a deterministic safe payoff `b_safe` or a short safe stream

The classical value of action `A` is

$$
Q^0(A) = b - p \gamma^{L-1} C.
$$

If action `B` has deterministic payoff `b_{\text{safe}}`, set

$$
Q^0(B) = b_{\text{safe}}.
$$

For classical indifference, choose

$$
b_{\text{safe}} = b - p \gamma^{L-1} C.
$$

Then sweep a small offset around this tie.

### Why this family is useful

This family produces two actions that can be classically tied but have very different local structure:

- one branch has a delayed low-probability catastrophe,
- the other is steady and safe.

This changes the local signed-margin geometry along the risky branch and creates a plausible setting in which the nonlinear operator can separate the actions.

### Parameters to sweep

- immediate bonus `b`
- catastrophe size `C`
- probability `p`
- delay `L`
- tie offset `ε`
- optional safe-stream shape if `B` is a stream rather than a single payoff

### Good variants

- deterministic warning state before catastrophe
- branch with a shallow early reward then latent risk
- multiple small catastrophe probabilities instead of a single event
- matched classical value but different return concentration

### Required outputs for this family

- classical versus safe phase diagram
- policy disagreement map
- start-state flip region
- clipping diagnostics and value-gap plot

---

## Concrete candidate family C: misaligned stress / safety family

This family exists to justify clipping.

### Template requirements

Construct a task where:

1. a raw weighted-LSE schedule with larger temperature enters a locally expansive or unstable region on visited states,
2. the safe clipped counterpart stays stable,
3. the environment and raw schedule are otherwise identical.

### How to force the stress regime

Design the visited branch so that, under the raw operator, a substantial portion of visited transitions satisfy

- negative or misaligned signed margin,
- local continuation derivative above the safe certificate,
- and preferably local derivative above 1 on some visited region.

Then compare:

- raw value iteration behavior,
- safe clipped value iteration behavior,
- raw versus safe fixed-policy evaluation,
- raw versus safe control.

### What to measure

- convergence or divergence rate
- oscillation amplitude
- residual norm trajectory
- fraction of visited transitions with `d_raw > 1`
- fraction with `d_raw > κ_t`
- safe-clipped derivative distribution
- whether the safe operator preserves any useful policy/value difference while preventing instability

### Important point

Do not make this family the main story. It is a supporting argument for why clipping is necessary.

---

## Required search algorithm

Implement the search in exact planning space before RL.

### Step 1: enumerate a family grid

For each family, generate a parameter grid over:

- horizon `T`
- branch lengths
- reward scales
- tie offsets
- shape parameters
- stochasticity parameters
- raw schedule proposals
- headroom / clipping parameters

### Step 2: find exact or near-exact classical tie locations

For each fixed geometry parameter `ψ`, find a near-tie `λ_tie(ψ)` such that the contest-state classical gap is near zero.

Store:

- `lambda_tie`
- contest state
- exact `Δ0`
- normalized gap
- occupancy of the contest state

### Step 3: evaluate a narrow neighborhood around the boundary

For each `(ψ, λ_tie + ε)` in a symmetric band of small offsets:

- solve classical exact control
- solve safe exact control
- solve raw exact control if relevant
- compute all search metrics

### Step 4: keep only decision-relevant candidates

Promotion requires nontrivial reachable policy/value differences under the **safe clipped operator**, not merely the raw operator.

### Step 5: export phase diagrams

For every family that contains shortlisted tasks, produce a plot over two parameters, for example `(λ, ψ)`, showing:

- sign of the classical action gap,
- sign of the safe action gap,
- classical decision boundary `Δ0 = 0`,
- safe decision boundary `Δ_safe = 0`,
- clip fraction or local derivative as an overlay.

This plot is one of the clearest possible demonstrations of the mechanism.

---

## Work package 0: consistency audit

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

## Work package 1: replace the task-search objective

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

5. **Classical tie diagnostics**
   - `contest_gap_abs = |Δ0(x_c)|`
   - `contest_gap_norm = |Δ0(x_c)| / reward_scale`
   - contest-state occupancy under `d_ref`

6. **Clipping diagnostics**
   - clip fraction on visited transitions
   - fraction of states/transitions where clipping is inactive
   - fraction where clipping saturates

7. **Safety stress diagnostics**
   - raw operator local derivative statistics
   - whether raw value iteration diverges, oscillates, or converges very slowly

### Initial promotion thresholds

Use these thresholds for the first search pass and do **not** relax them post hoc after looking at final RL outcomes.

A task is promotable only if all of the following hold:

1. `policy_disagreement >= 0.05` under `d_ref` **or** the start-state greedy action differs.
2. `mass_delta_d >= 0.10`.
3. normalized `|value_gap| >= 0.005`.
4. designated contest state has `contest_gap_norm <= 0.01`, or at minimum `<= 0.02` with explicit justification.
5. designated contest state is reachable: occupancy under `d_ref >= 0.05`, unless the start-state action differs.
6. clip fraction is in a nondegenerate band, for example between `0.05` and `0.80` on `d_ref` mass.
7. no certification violation occurs.

If too few tasks pass, report that outcome honestly. Do **not** silently move thresholds.

### Deliverables

- `results/search/candidate_grid.parquet`
- `results/search/candidate_metrics.parquet`
- `results/search/near_indifference_catalog.csv`
- `results/search/shortlist.csv`
- `results/search/shortlist_report.md`
- `results/search/phase_diagram_data.parquet`

---

## Work package 2: design better candidate task families

Do **not** just shrink `R_max` and rerun the old dense-cost chains. Smaller `R_max` can enlarge the certified cap, but it can also shrink the key `(r - V)` factor and leave policy differences at zero.

Instead, construct candidate families where the classical decision boundary is intentionally close to indifferent and the nonlinear operator can plausibly flip the preferred action.

### Required candidate families

#### Family A: aligned delayed-reward propagation

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
- branch shape parameter `ψ`

#### Family B: policy-flip control tasks

Goal: produce tasks where the nonlinear Bellman target changes the optimal action at one or more reachable states.

Design requirements:
- at least one reachable state with classical action gap near zero
- nonlinearity should favor one branch through the signed-margin mechanism
- the flip should survive safe clipping

A good pattern is a forked chain or forked grid where:
- branch 1 has slightly better classical continuation,
- branch 2 has stage-local structure that the safe operator amplifies,
- the classical action gap is small enough that the nonlinear correction can matter.

#### Family C: misaligned stress / safety tasks

Goal: show why safe clipping is needed.

Design requirements:
- a raw weighted-LSE configuration that enters an expansive or unstable regime,
- a safe clipped counterpart that remains stable,
- identical environment and raw schedule so the difference is attributable to clipping.

Measure instability explicitly rather than just showing a noisy return curve.

### Important design principle

For at least some tasks, tune the environment so that the **optimal policy itself depends on the operator nonlinearity**, not just the logged activation metrics.

---

## Work package 3: change the planning experiments

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
5. **Decision-boundary phase diagram:** classical and safe action-boundary curves on the same parameter plane.

### Deliverables

- `results/planning/limited_backup_metrics.parquet`
- `figures/main/fig_decision_boundary_phase_diagram.pdf`
- `figures/main/fig_propagation_curve.pdf`
- `figures/main/fig_stagewise_error_heatmap.pdf`
- `figures/main/fig_safe_vs_raw_stability.pdf`

---

## Work package 4: simplify and strengthen the baselines

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

## Work package 6: statistical reporting

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
2. **Decision-boundary figure:** classical and safe boundaries on the same family parameter plane.
3. **Propagation figure:** safe vs classical under limited backups on an aligned task.
4. **Safety figure:** raw vs clipped behavior on a stress task.
5. **RL translation figure:** only for shortlisted tasks that exact planning says are nontrivial.
6. **Compact summary table:** shortlist metrics, baselines, and main outcomes.

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
9. **Do not** search only over reward scale. Search over both classical-tie parameters and geometry parameters.
10. **Do not** claim a policy-flip task is meaningful unless the flipped state is reachable.

---

## Two allowed end states

### End state A: positive mechanism-backed paper

You may claim a positive empirical story only if the final shortlisted tasks show:
- clear exact policy disagreement,
- nontrivial exact value translation,
- a mechanism-level planning advantage or safe-vs-raw stability result,
- and RL translation that is larger than noise under adequate seeds.

If that happens, the paper should be framed as:
- a theory paper with targeted mechanism experiments,
- not as a broad benchmark-improvement paper.

### End state B: theory + negative-result paper

If the search produces activation without policy/outcome translation, or policy translation without robust RL gains, say so clearly.

Then frame the paper as:
- certification and safe clipping are correct,
- expressive nonlinear behavior can be activated,
- but worst-case-safe calibration suppresses practical expressivity on broad reward scales,
- and identifying nontrivial policy-regime tasks remains open.

That is a valid and honest contribution. It is stronger than overstating null benchmark results.

---

## Literal implementation checklist

Run the following in order:

1. audit current tables and figures
2. fix any consistency failure
3. implement family generators with designated contest states
4. implement exact classical tie solver `λ_tie(ψ)`
5. compute near-indifference catalogs
6. generate classical/safe phase diagrams
7. compute exact shortlist metrics
8. select all tasks meeting the predeclared thresholds
9. run limited-backup planning diagnostics on shortlisted tasks
10. run safety-stress comparison on at least one stress family
11. run RL only on shortlisted tasks
12. compile main-paper figures from the shortlisted tasks only
13. write down whether the result is a positive mechanism story or a negative-result story

---

## Final instruction

The single most important change is this:

> **Stop asking “Where does safe beat classical on return?”**
>
> **Start asking “Where does the theorem imply a nontrivial policy/value difference, and does that difference survive safe clipping and show up in planning or RL?”**

All experiment design, task search, metric selection, and plotting must be aligned to that question.
