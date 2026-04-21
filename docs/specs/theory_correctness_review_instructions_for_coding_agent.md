# Theory-Correctness Review Follow-Up for the Coding Agent

This document is for the coding agent.

It is **not** a mandatory repair spec and it is **not** an instruction to implement every item below. Treat it as a theory-audit handoff: review the cited paper/specs/code, assess whether each issue is still present in the current tree, decide what matters scientifically, and then choose the appropriate next action.

Your job is to **assess, decide, and justify**. Do not mechanically apply this memo.

---

## 1. Purpose

A review of the current codebase against the TAB paper and the phase specs found several likely mismatches between:

1. the theory in the paper,
2. the repository's own spec documents,
3. the current implementation.

Some issues appear strong and likely real. Some may already be fixed in current code, may be partially stale, or may only matter under certain task assumptions.

Before changing code, determine:

1. whether the issue is real in the current tree;
2. whether it materially affects scientific conclusions;
3. which source is authoritative when the paper, older audits, and current code disagree;
4. whether the right response is:
   - fix now,
   - add tests first,
   - defer and document,
   - or conclude no change is needed.

---

## 2. Primary sources to consult

Review these before making decisions:

### Paper

- `paper/TAB_Information_Geometric/TAB_information_geometric_rewrite.pdf`
- `paper/TAB_Information_Geometric/TAB_information_geometric_rewrite.tex`

### Existing agent/audit docs

- `docs/specs/phase_IV_A_repair_instructions_for_coding_agent.md`
- `docs/specs/numerical_formula_audit_TAB_code.md`
- `docs/specs/refined_phase_I_II_III_code_audit.md`

### Phase specs

- `docs/specs/phase_I_classical_beta0_experiments.md`
- `docs/specs/phase_II_stress_test_beta0_experiments.md`
- `docs/specs/phase_III_safe_weighted_lse_experiments.md`
- `docs/specs/phase_IV_A_activation_audit_and_counterfactual.md`
- `docs/specs/phase_IV_B_translation_experiments.md`
- `docs/specs/phase_IV_C_advanced_stabilization_and_geometry_ablations.md`

---

## 3. Operating stance

When working through the items below:

1. Do **not** assume an older audit is automatically correct.
2. Do **not** assume passing tests imply theory-correctness.
3. Do **not** assume a docstring reflects the actual implemented math.
4. Prefer the strongest available evidence:
   - current code behavior,
   - direct comparison to the paper equations,
   - direct comparison to the repo's own phase specs,
   - focused tests or synthetic witnesses.
5. If a finding is ambiguous, say so explicitly and resolve it before making broad edits.

---

## 4. Issues to assess

The items below are intentionally phrased as **assessment targets**, not directives.

### A. Phase IV-A natural-coordinate calibration

Assess whether the current Phase IV-A scheduler still violates the natural-coordinate construction in the paper/specs by using `R_max` where the deployed geometry should use `A_t = R_max + Bhat[t+1]`.

Relevant files:

- `experiments/weighted_lse_dp/geometry/phase4_calibration_v3.py`
- `experiments/weighted_lse_dp/geometry/adaptive_headroom.py`
- `docs/specs/phase_IV_A_activation_audit_and_counterfactual.md`
- `docs/specs/phase_IV_A_repair_instructions_for_coding_agent.md`

Questions to answer:

1. Is `xi_ref_t` currently defined from `a_t = sign_family * margin / A_t`, or from `margin / R_max`, or from a mixed approximation?
2. If `A_t` changes during fixed-point headroom iterations, is `xi_ref_t` recomputed accordingly?
3. Are `u_ref_used_t`, `theta_used_t`, and `beta_used_t` internally consistent with the natural-coordinate identities?
4. If the implementation is not faithful, is this likely large enough to explain the current design-point versus replay gap?

### B. Phase IV-A finite-horizon stagewise pilot

Assess whether the Phase IV-A pilot is using the correct finite-horizon continuation value.

Relevant files:

- `experiments/weighted_lse_dp/geometry/task_activation_search.py`
- `experiments/weighted_lse_dp/runners/run_phase4_counterfactual_replay.py`
- `docs/specs/phase_IV_A_repair_instructions_for_coding_agent.md`

Questions to answer:

1. Does the pilot compute a full `V[t, s]` table, or only a single stage-collapsed `V_star[s]`?
2. Are logged margins using `r_t - V[t+1, s_next]` as the finite-horizon spec requires?
3. If not, how much could this alter sign selection, `p_align`, `xi_ref`, and replay eligibility?
4. Is the current pilot acceptable for any subset of tasks, or is it fundamentally misaligned with the paper/spec?

### C. Phase III representative aligned margin

Assess whether the Phase III schedule builder is using the correct statistic for the representative aligned margin.

Relevant files:

- `experiments/weighted_lse_dp/calibration/build_schedule_from_phase12.py`
- `experiments/weighted_lse_dp/calibration/calibration_utils.py`
- `experiments/weighted_lse_dp/common/calibration.py`
- `docs/specs/phase_III_safe_weighted_lse_experiments.md`

Questions to answer:

1. Does current code use the spec's `Q_0.75(a_t | a_t > 0)` with `a_t = s * m_t`?
2. Or does it use a conditional mean, raw positive margins, or another proxy?
3. For negative-sign families, are the logged stagewise statistics actually sign-aligned, or only raw-margin positive/negative splits?
4. If the code is using a proxy, is that an acceptable approximation, or does it materially change the schedule?

### D. ExpectedSARSA continuation logging and calibration semantics

Assess whether `v_next_beta0`, `margin_beta0`, and derived calibration statistics are consistent with the actual bootstrap used by `ExpectedSARSA` and `SafeExpectedSARSA`.

Relevant files:

- `experiments/weighted_lse_dp/common/callbacks.py`
- `mushroom-rl-dev/mushroom_rl/algorithms/value/td/safe_expected_sarsa.py`
- `docs/specs/phase_I_classical_beta0_experiments.md`
- `docs/specs/phase_IV_A_activation_audit_and_counterfactual.md`

Questions to answer:

1. Does the transition logger record greedy `max_a Q(s',a)` even for ExpectedSARSA-style methods?
2. If yes, which downstream artifacts rely on those logged values?
3. Is the mismatch only a logging/provenance problem, or does it bias calibration/schedule construction in a scientifically meaningful way?
4. Should the fix be algorithm-specific logging, separate schemas, or explicit exclusions from calibration?

### E. Regime-shift adaptation "AUC" semantics

Assess whether the current adaptation metrics are consistent with the spec's intended AUC language.

Relevant files:

- `experiments/weighted_lse_dp/common/callbacks.py`
- `docs/specs/phase_II_stress_test_beta0_experiments.md`

Questions to answer:

1. Is `pre_change_auc` / `post_change_auc` currently an actual area-under-curve quantity, or just a mean?
2. If it is just a mean, is that a naming issue, a metric-definition bug, or an acceptable simplification that should be documented?
3. Would changing it break existing processed results or only improve semantic correctness?

### F. Core safe DP Bellman operator

Assess the current state of the older claim that safe DP planners compute `g(r_bar, E[V'])` instead of `E[g(r_bar, V')]`.

Relevant files:

- `mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_value_iteration.py`
- `mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_policy_evaluation.py`
- `mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_modified_policy_iteration.py`
- `mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_async_value_iteration.py`
- `mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_common.py`
- `docs/specs/numerical_formula_audit_TAB_code.md`

Questions to answer:

1. Is the major operator bug from the older audit still present in the active code paths?
2. If not, what exactly was fixed?
3. Are any docstrings, helper functions, or secondary code paths still inconsistent with the paper and likely to mislead future work?

### G. Resolved or possibly stale older findings

Actively verify whether the following older findings are still live or already fixed:

1. tiny-`beta` threshold handling in the scalar operator;
2. `metrics.aggregate(..., axis=1)` bootstrap CI shape bug;
3. empty-input tail-risk crash;
4. PE / SafePE reference-policy construction for grid/taxi tasks.

Relevant files:

- `mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_common.py`
- `experiments/weighted_lse_dp/common/metrics.py`
- `experiments/weighted_lse_dp/common/callbacks.py`
- `experiments/weighted_lse_dp/common/task_factories.py`
- `experiments/weighted_lse_dp/runners/run_phase2_dp.py`
- `experiments/weighted_lse_dp/runners/run_phase3_dp.py`

Do not carry forward stale conclusions without re-checking current code.

---

## 5. Decision prompts

For each issue you assess, decide explicitly:

1. **Status**
   - confirmed current bug,
   - likely current bug but needs runtime proof,
   - stale / already fixed,
   - spec mismatch but low impact,
   - or acceptable implementation choice.

2. **Scientific severity**
   - could change paper-level conclusions,
   - could change intermediate calibration/schedule quantities only,
   - mostly provenance/metric naming,
   - or negligible.

3. **Best next action**
   - fix immediately,
   - write tests first,
   - document and defer,
   - or no change.

4. **Validation burden**
   - what must be rerun if fixed:
     - unit tests only,
     - Phase IV-A gate only,
     - Phase II/III calibration regeneration,
     - Phase IV-B/C blocked pending repair,
     - etc.

---

## 6. Preferred deliverable format

Produce a short decision memo before or alongside implementation with sections like:

```text
Issue
Current status
Why it matters
Evidence
Decision
If fixing now: validation / rerun plan
If not fixing now: rationale
```

If multiple issues are related, group them by subsystem:

- core operator,
- calibration/schedule,
- Phase IV-A pilot/replay,
- logging/metrics,
- experiment runners.

---

## 7. Important constraint

This memo is meant to improve judgment, not remove it.

Do **not** read it as:

- "all listed issues are definitely bugs,"
- "all of them must be fixed now,"
- or "older audits override current code inspection."

Instead, use it to run a careful theory-correctness pass and make explicit, defensible decisions.
