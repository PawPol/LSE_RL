# Reply to the Coding Agent: Proceed with Phase V (Positive-Only Empirical Plan)

Your redesign direction is broadly correct. The main thing now is to **stop blocking on clarifications** and execute the mechanism-first rebuild.

There is one major change to the previous handoff:

> **Negative-result framing is not allowed.**

The paper must converge to a **positive, scoped, mechanism-backed empirical story**. That does **not** mean broad benchmark superiority. It means you must build a small set of theorem-linked constructed tasks where the **deployed safe clipped operator** shows clear policy/value translation, propagation effects, and a clean safe-vs-raw contrast.

Proceed under the following decisions.

## Decisions on your six questions

### 1) Naming: Phase V or Phase IV-D?
Use **Phase V**.

Reason: this is not a patch to the old empirical program. The search objective, promotion criteria, and main-paper story are all changing. Keep Phase I–IV intact as appendix sanity material and build the replacement empirical section under a clean new top-level name.

Use:
- `docs/specs/phase_V_mechanism_experiments.md`
- `results/search/...`
- `results/planning/...`
- `results/rl/...`
- `figures/main/...`

Do not spend time debating the label further.

### 2) What if the shortlist is empty after honest screening?
Do **not** pivot to a negative-result paper.

If the shortlist is empty, or if the first family pass produces activation without enough policy/value translation, the default action is:

1. keep the thresholds fixed,
2. refine the candidate families,
3. trace classical and safe decision boundaries,
4. choose new interior parameter points where the **deployed clipped safe operator** actually changes the preferred action,
5. and rerun exact planning.

In other words: **change family design, not thresholds, and do not reframe the paper as a null result**.

Concretely:
- keep Family A/B/C,
- refine A and B first,
- add D/E if needed,
- continue until you have at least:
  - **2 promotable positive exact-planning families**, and
  - **1 safety/stability family**.

Do not ask again whether a negative-result pivot is acceptable. It is not.

### 3) Compute budget
Use a **two-stage budget**.

#### Stage 1: pilot
- Run exact search/planning first.
- For RL, run a **5-seed paired pilot** only on tasks that already show exact policy/value translation.
- Include the required arms only:
  - classical `beta=0`
  - safe-zero
  - safe-nonlinear
  - tuned fixed-discount baseline
  - multi-step baseline
  - raw-unclipped only on safety-stress tasks

#### Stage 2: final
- Promote at most **3 tasks** to the final RL set.
- Run **20 paired seeds** on those tasks.
- Use common randomness / paired seeds across arms.

If the pilot is weak, do **not** write a null-result summary. Instead, go back to family design and exact-planning search. Exact-planning evidence carries the main empirical story; RL is a follow-on confirmation layer.

### 4) Raw-operator runs / safety behavior
Use a **hard iteration cap plus NaN/overflow/divergence guards**.

Do not allow uncontrolled runaway jobs.

For raw runs, log one of the following outcomes explicitly:
- converged,
- oscillatory,
- expansive/divergent,
- overflow/NaN guarded stop,
- failed to reach tolerance by iteration cap.

For Family C, divergence or oscillation is itself a valid **supporting** result. The point is to show that clipping restores stability while preserving as much useful nonlinear effect as possible.

### 5) Alpha sweep currently in flight
Treat the existing alpha-sweep work as **appendix sanity material only**.

Rules:
- If it is already mostly complete, keep it and archive it under appendix / sanity.
- If it is incomplete or interferes with Phase V, stop investing in it.
- Do not let it block or redefine the new empirical story.
- Do not use it as a reason to keep Phase IV-style activation-first search alive.

### 6) Spec document location
Write the canonical spec to:

`docs/specs/phase_V_mechanism_experiments.md`

Use that as the single source of truth. Do not split the redesign across ad hoc notes or mixed Phase IV/Phase V filenames.

---

## Additional execution directives

### A. Positive-only end state
The only acceptable empirical endpoint is a **positive, scoped, mechanism-backed paper**.

That means the final evidence should show:

- a theorem-linked family where the deployed safe clipped operator changes the optimal action or exact start-state value relative to classical discounting,
- a clear propagation or stagewise-error advantage in exact planning,
- a safe-vs-raw stability contrast,
- and at least one RL task where the translation survives beyond noise.

Do **not** aim for broad generic benchmark dominance.

Do **not** write a negative-result paper.

### B. If the first family pass is weak, do this next
Do **not** change thresholds and do **not** stop.

Instead, iterate the task design using the following levers:

1. **Classical tie control**
   - solve `lambda_tie(psi)` so the classical action gap is near zero at a designated reachable contest state.

2. **Geometry control**
   - introduce a second parameter that changes the temporal shape of reward / warning / catastrophe exposure while leaving the classical value nearly unchanged.

3. **Boundary tracing**
   - compute both the classical decision boundary and the safe decision boundary,
   - then choose interior points where the safe clipped operator flips the preferred action.

4. **Propagation amplification**
   - increase delay,
   - increase horizon,
   - sharpen reward concentration,
   - deepen warning structure,
   - or increase branch asymmetry,
   - always within certified bounds.

5. **Family expansion order**
   - refine **Family A** first,
   - refine **Family B** second,
   - keep **Family C** for safe-vs-raw stability,
   - add **Family D** for matched-classical-value but different temporal concentration / revelation,
   - add **Family E** for regime-shift / warning revelation tasks.

The job is not “search until anything wins.”  
The job is “engineer a theorem-linked family where the clipped operator should differ, then verify that it does.”

### C. First deliverables
Do the following first, in order:

1. **WP0** consistency audit and fail-loud gates.
2. **Phase V spec** at `docs/specs/phase_V_mechanism_experiments.md`.
3. **WP1 metric/search implementation** including tie solver, `d_ref`, policy disagreement, value gap, `delta_d`, clipping diagnostics.
4. **Family A / B / C task factories**.
5. exact-search shortlist generation.
6. if the shortlist is weak, refine families and add D/E.
7. limited-backup planning diagnostics.
8. RL pilot only after shortlist exists.

### D. The prefilter and shortlist thresholds are fixed
Use the thresholds from the instruction file. Do not relax them post hoc.

#### Near-tie prefilter
For each designated reachable contest state:
- strict near-tie band: `gap_norm <= 0.01`
- exploratory soft band: `gap_norm <= 0.02`

Use the soft band only for exploration. Final promotion requires a genuine near-tie or a clearly justified equivalent.

Reachability condition:
- contest-state occupancy under `d_ref` at least `0.05`, **or**
- start-state greedy action differs.

#### Promotion thresholds
A task is promotable only if all of the following hold:

1. `policy_disagreement >= 0.05` under `d_ref` **or** the start-state greedy action differs.
2. `mass_delta_d >= 0.10`.
3. normalized `|value_gap| >= 0.005`.
4. clip fraction lies in a nondegenerate band, e.g. `0.05 <= clip_fraction <= 0.80` on `d_ref` mass.
5. no certification violation occurs.

If too few tasks pass, report that outcome internally and continue constructive family design. Do not silently move thresholds.

### E. Explicit metrics to compute for every candidate
For each exact-planning candidate, compute and save:

- `margin_pos = E_{d_ref}[max(beta_tilde_t * (r - V_next_ref), 0)]`
- normalized `margin_pos`
- `delta_d = E_{d_ref}[|d_safe_t - gamma|]`
- `mass_delta_d`
- `policy_disagreement`
- start-state greedy-action-flip flag
- `value_gap = V_safe_0(s0) - V_classical_0(s0)`
- normalized `value_gap`
- clip fraction / inactive fraction / saturation fraction
- raw-operator local derivative stats
- raw-run convergence/divergence status where applicable

### F. Required baselines are not optional
Main baselines:
- classical `beta=0`
- safe-zero
- tuned fixed-discount baseline
- multi-step baseline (`n`-step or `TD(lambda)`)
- raw-unclipped weighted-LSE only for safety/stability tests

Do not pad the paper with a broad menu of weakly informative arms.

### G. Main-paper figure budget
Target the main paper to **5 figures and 1 table**.

Recommended main-text figures:
1. mechanism frontier / phase-diagram shortlist figure,
2. propagation figure,
3. stagewise error heatmap,
4. safe-vs-raw stability figure,
5. RL translation + mechanism diagnostics,

and one compact summary table.

Phase I–IV results belong in the appendix unless directly needed as sanity checks.

### H. Manifest and reproducibility
Generate the experiment manifest from the runner, not post hoc.

It must include:
- git SHA,
- exact argv,
- seeds,
- task config,
- calibration config,
- timestamp,
- output paths.

Write it to:

`results/summaries/experiment_manifest.json`

---

## One correction to your current behavior
Do not frame the next step as “waiting for answers” before you can spawn the planner.

The next step is:
1. write the Phase V spec,
2. materialize the task checklist,
3. start WP0 and WP1.

That is the default action now.

---

## Final instruction
Keep the paper aligned to this question:

> Where does the theorem imply a nontrivial policy/value difference, does that difference survive safe clipping, and does it show up in planning or RL?

Do not revert to activation-first search, broad null benchmark matrices, post hoc threshold movement, or negative-result framing.
