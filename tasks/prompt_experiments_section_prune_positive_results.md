# Prompt for coding agent: prune experiments to the strongest positive NeurIPS story

You are editing the new paper version, especially `paper/neurips_selective_temporal_credit_assignment_positioned.tex` and the figures/tables used in Section 14.

The current experiments section is too detailed and contains many internal Phase V/VI diagnostics, null-family searches, ablation histories, and plots that are either weak or distracting. For a NeurIPS theory-style submission, the main paper should keep only the strongest positive evidence that directly supports the operator theory.

## Core instruction

Rewrite the experiments section so that the main text is built around **only two positive empirical claims**:

1. **Family A: the safe weighted-LSE objective selects a better policy under the safe objective while incurring essentially zero classical-objective regret.**
2. **Family C: safe clipping preserves the contraction certificate where the raw weighted-LSE operator violates it.**

Everything else should be removed from the main experimental narrative. Do not include broad Phase I–IV benchmark matrices, activation-follow-up null results, Family B/D/E failure searches, long ablation tables, or weak propagation plots in the main text.

Do **not** run new experiments. This is a paper-editing and plot-selection task. Use the already generated results.

---

## Keep in the main text

### 1. One compact summary table

Keep a small table with only the headline rows that support the two positive claims. The table should be readable to a reviewer without knowing internal phase names.

Recommended rows:

| Claim | Family | Evidence | Headline numbers |
|---|---|---|---|
| Objective-level policy improvement | A deterministic | Safe policy flips the start-state decision at near-classical ties | `n=5`, start-state flip rate `1.0`, max `|value_gap_norm| = 7.82e-3`, max policy disagreement `0.25`, clipping mode `safe_active_no_distortion` |
| Dual-objective Pareto improvement | A stochastic | Safe policy has zero classical-eval regret but positive safe-eval advantage | `n=6`, classical-eval regret `<= 4.4e-16`, safe-eval advantage from about `0.012` to `0.499`, mean safe advantage about `0.156`, flip rate `1.0` |
| Certified stability | C raw-stress | Safe-clipped operator stays inside cert; raw operator violates it | `n=84`, safe derivative bounded by `kappa_t in [0.95, 0.96]`, raw draw up to about `1.416`, clipping mode `binding_clip` |

Avoid internal labels such as `Phase VI-E`, `Phase VI-C`, or `translation_4a2`. Use clean language: “deterministic Family A,” “stochastic Family A,” and “Family C raw-stress.”

### 2. Main Figure A: Dual-objective Family A result

Use `fig_dual_objective_eval.pdf` or regenerate a cleaner version from the same data.

The figure must communicate this message:

\[
V^{\pi^*_{cl}}_{cl}(s_0) - V^{\pi^*_{safe}}_{cl}(s_0) \approx 0,
\qquad
V^{\pi^*_{safe}}_{safe}(s_0) - V^{\pi^*_{cl}}_{safe}(s_0) > 0.
\]

The current figure contains the right result, but the x-axis is dominated by machine-precision noise around `1e-16` and the right panel’s four grouped bars on log scale obscure the main claim. Prefer a cleaner plot if possible:

- Panel A: safe-eval advantage for the six stochastic Family A tasks, sorted by reward scale or task id.
- Panel B: classical-eval regret on the same tasks, with a clear annotation such as “all regrets <= 4.4e-16.”

Alternative acceptable plot:

- Keep the Pareto scatter, but make the x-axis visually clear by labeling it “classical-eval regret, numerically zero” and avoid making `1e-16` look like a meaningful continuum.
- Replace the log-scale grouped-bar panel with a direct paired-difference panel showing safe-objective advantage.

Caption should say, in plain language:

> On stochastic Family A, the safe-optimal policy is indistinguishable from the classical-optimal policy under the classical objective, but strictly better under the safe weighted-LSE objective on every task. This is an objective-level Pareto improvement, not a noisy RL return effect.

### 3. Main Figure B: Family C safe-vs-raw stability

Keep `fig_safe_vs_raw_stability.pdf`, possibly with cosmetic relabeling.

This is a strong and important plot. It shows exactly why the safe clipped operator is necessary:

- the raw operator gives a much larger value trajectory on the lead task (`V[0,s0]` about `16.9`);
- classical and safe remain close and stable (`V[0,s0]` about `4.86` and `4.92`);
- safe local continuation derivatives stay below the certified bound `kappa_t`;
- raw local continuation derivatives cross `1`, breaching the contraction certificate.

Caption should emphasize:

> Safe clipping preserves the theorem’s contraction certificate on visited transitions, while the raw operator leaves the certified region. The point is not that finite-horizon raw value iteration always diverges; the point is that the raw operator loses the guarantee that supports the safe DP/RL toolbox.

---

## Remove from the main text

Remove the following from Section 14 and from any main-paper result table:

### 1. Family B, D, and E null searches

Do not discuss that B/D/E promoted zero tasks in the main text. Do not include their candidate counts, failure explanations, or structural “sequential chains do not translate” discussion.

Those details make the empirical section look like a failed search process. They can be omitted entirely or reduced to one appendix sentence, but they should not appear in the main NeurIPS narrative.

### 2. Phase I–IV benchmark matrix and activation follow-up

Do not include broad Phase I–IV results in the main text. Remove or demote:

- eight-task Phase I–III matrix;
- calibration sanity tables;
- ablation suites;
- dense-cost activation follow-up;
- matched-control null RL translation;
- consistency-audit details.

The earlier experiments are useful internally, but they do not support the main positive claim. If they remain anywhere, they should be in a short appendix note or supplementary material, not as a main result.

### 3. Weak propagation diagnostics

Remove these figures from the main text:

- `fig_propagation_curve.pdf`
- `fig_stagewise_error_heatmap.pdf`
- `fig_greedy_recovery.pdf`

Reason: they do not show a large, visually compelling improvement. The propagation/greeedy-recovery result is at most a one-backup or small-percentage advantage, and the heatmaps look almost identical. These figures dilute the strongest message.

If you keep them at all, move them to an appendix as optional diagnostics. They should not be part of the main empirical story.

### 4. Perturbation-growth figure

Remove `fig_perturbation_growth.pdf` from the main text.

Reason: it is nuanced rather than strongly positive. It shows raw perturbations behaving erratically, but the caption currently says all three operators remain bounded at finite horizon. That weakens the safety story if placed in the main section. The main safety figure should be `fig_safe_vs_raw_stability.pdf`, which directly shows certificate preservation versus certificate breach.

### 5. Risk-profile / CVaR section

Remove the risk-profile section from the main text. It says the optimistic Family A safe policy has similar mean, higher variance, and worse CVaR than the classical policy. That is technically understandable, but it distracts from the positive objective-level result.

If this material remains, it should appear only as a short caveat in the appendix: the sign of beta determines whether the operator is optimistic or pessimistic. Do not frame the main experiments around variance or CVaR.

---

## Replace Section 14 with this structure

Use this outline.

### 14 Experiments

Opening paragraph:

- Say the experiments are mechanism demonstrations for the Bellman operator, not broad benchmarking.
- State the two questions:
  1. Can the deployed safe weighted-LSE operator change the optimal policy at a classical near-tie while preserving classical-objective performance?
  2. Does safe clipping preserve the contraction certificate where the raw operator would leave the certified region?
- Say all experiments use exact finite-horizon planning/evaluation on constructed MDPs aligned with the theorems.
- Avoid internal phase names.

### 14.1 Near-tie construction and metrics

Keep this short.

Define:

- classical action gap at a contest state;
- policy disagreement or start-state flip;
- normalized safe value gap;
- local continuation derivative `d_t(r,V)`;
- clipping modes `safe_active_no_distortion` and `binding_clip`.

Do not enumerate all searched families. Do not include zero-promotion families.

### 14.2 Family A: objective-level policy improvement at classical ties

Core text:

- Explain Family A as delayed jackpot versus smooth stream / concentration contrast.
- The classical tie is intentionally tuned so the two policies are essentially equal under classical evaluation.
- The safe operator, through the weighted-LSE objective, chooses the delayed-jackpot/concentrated-signal branch.
- Deterministic Family A: start-state flips on all 5 promoted tasks; max normalized value gap `7.82e-3`; max policy disagreement `0.25`; clipping inactive because the raw schedule is already certified.
- Stochastic Family A: under classical evaluation the safe policy has regret no larger than machine precision (`<= 4.4e-16`), while under safe evaluation it strictly dominates on every task, with safe-eval gap from about `0.012` to `0.499` and mean about `0.156`.

Use the dual-objective figure here.

Suggested claim language:

> This is the cleanest empirical signature of the proposed operator: at a classical near-tie, the safe weighted-LSE objective selects a policy that is classically regret-free but strictly preferred by the safe recursive objective.

### 14.3 Family C: clipping preserves the certificate that raw violates

Core text:

- Explain Family C as a raw-stress family designed to make the raw weighted-LSE derivative exceed the certified contraction budget.
- On the 84 promoted safety tasks, the raw operator violates the local derivative certificate; the safe operator clips the schedule and remains inside the bound.
- On the lead task, raw reaches `V[0,s0] ≈ 16.9`, while safe and classical remain stable around `4.92` and `4.86`.
- Safe derivative mass lies below `kappa_t`, while raw places mass beyond `d_t=1`.

Use the safe-vs-raw stability figure here.

Suggested claim language:

> The raw operator is the expressive object; the clipped operator is the deployable object. Family C shows why the distinction matters: clipping preserves the contraction certificate without collapsing back to a purely classical backup.

### 14.4 Takeaway

Short, one paragraph.

Say:

- Family A demonstrates positive mechanism translation at a tuned classical decision boundary.
- Family C demonstrates that the safety layer is necessary and effective.
- These experiments support the theory: the weighted-LSE operator is not a universal return-improvement heuristic; it is a certified Bellman-objective mechanism that changes temporal credit assignment in theorem-linked cases.

Do not include long caveats, null results, or audit details here.

---

## Figure and caption cleanup

Update figure names/captions so they sound like final paper results, not internal engineering runs.

Avoid titles containing:

- `Phase V`
- `Phase VI`
- `A_003`
- `family_C_05_L4_Rp1_mult128` in the title line
- `translation_4a2`
- `smoke`
- `shortlist`

Task IDs may appear in small annotations or captions if needed, but not as the main figure title.

Recommended figure names:

- `fig_familyA_dual_objective.pdf`
- `fig_familyC_certificate.pdf`

Recommended captions:

**Family A dual objective.**
“Safe weighted-LSE policy selection at a classical near-tie. Across six stochastic Family A tasks, the safe-optimal policy has machine-precision classical-evaluation regret and strictly positive safe-evaluation advantage. The mechanism therefore lives in the Bellman objective rather than in stochastic return noise.”

**Family C certificate.**
“Safe clipping preserves the contraction certificate. On the raw-stress task, the raw weighted-LSE operator places derivative mass beyond the nonexpansive threshold, while the deployed safe operator remains below the certified stagewise bound. The safe values remain close to classical, whereas the raw target leaves the certified region.”

---

## Writing constraints

- Keep the main experiments section to about 1 to 1.5 pages if possible.
- Use only one compact table and two main figures.
- Do not overclaim benchmark superiority.
- Do not use “negative result” language.
- Do not describe failed family searches in the main paper.
- Do not include RL curves unless they show a clear positive result. The current positive story is exact planning/evaluation, not RL learning curves.
- Do not round small quantities to `0.0000`; use scientific notation.
- Compile the paper and fix all stale references after deleting figures/tables.
- Ensure every remaining experiment claim is supported by a corresponding result artifact and figure/table.

---

## Acceptance criteria

The edit is successful if the experiments section communicates the following in under two pages:

1. **Positive mechanism result:** safe weighted-LSE chooses a policy that is classically regret-free and strictly better under the safe objective on Family A.
2. **Positive safety result:** safe clipping preserves the certified derivative/contraction bound where raw weighted-LSE violates it on Family C.
3. **No clutter:** failed searches, broad calibration matrices, weak propagation plots, CVaR caveats, and audit details are absent from the main experimental story.
4. **No internal phase jargon:** the final paper reads like a clean NeurIPS submission, not a lab notebook.
