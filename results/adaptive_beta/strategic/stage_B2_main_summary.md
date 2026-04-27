# Phase VII-B Stage B2-Main — Summary Memo

**Branch:** `phase-VII-B-strategic-2026-04-26`
**Verdict (§17):** **NO UPDATE**

**Verdict reasoning:**
- fixed_negative ≥ adaptive_beta on auc_full in: ['hypothesis_testing']

---

## 1. Run Matrix

- 1 game (shapley) × 2 adversaries × 5 methods × 10 seeds × 10,000 episodes = **100 cells**.
- Manifest verified clean: all `status=completed`, no duplicates, no NaN, no divergence.
- Bootstrap: 10,000 paired resamples, percentile 95% CIs, paired by `seed_id`, fixed seed 0xB2DEF.

- Total divergence events across all 100 runs: **0**
- Total NaN counts: **0**

## 2. Final-Window Mean ± Std (mean over last 100 episodes)

| adversary | method | mean | std | n_seeds |
|---|---|---|---|---|
| FM-RM | vanilla | +12.075 | 0.205 | 10 |
| FM-RM | fixed_positive | +7.496 | 0.630 | 10 |
| FM-RM | fixed_negative | +12.178 | 0.206 | 10 |
| FM-RM | adaptive_beta | +12.052 | 0.216 | 10 |
| FM-RM | adaptive_sign_only | +12.027 | 0.154 | 10 |
| HypTest | vanilla | +8.776 | 0.314 | 10 |
| HypTest | fixed_positive | +7.350 | 0.366 | 10 |
| HypTest | fixed_negative | +9.143 | 0.295 | 10 |
| HypTest | adaptive_beta | +9.101 | 0.514 | 10 |
| HypTest | adaptive_sign_only | +8.954 | 0.488 | 10 |

## 3. Paired-Bootstrap Diffs vs Vanilla (10k resamples, 95% CI)

Sample-efficiency endpoint = `auc_first_2k`; secondary endpoints = `auc_full`, `final_return`, `recovery_time`.

### FM-RM

| method | metric | mean ± std | 95% CI | CI excl. 0 |
|---|---|---|---|---|
| fixed_positive | AUC₀₋₂ₖ | -1503.300 ± 180.306 | [-1608.400, -1397.400] | **yes** |
| fixed_positive | AUC₀₋₁₀ₖ | -32438.300 ± 1860.449 | [-33622.918, -31434.880] | **yes** |
| fixed_positive | final return | -4.579 ± 0.545 | [-4.936, -4.303] | **yes** |
| fixed_positive | recovery_time | -2606.500 ± 930.774 | [-2969.100, -1995.197] | **yes** |
| fixed_negative | AUC₀₋₂ₖ | -17.000 ± 191.658 | [-122.602, +101.805] | no |
| fixed_negative | AUC₀₋₁₀ₖ | +152.900 ± 335.880 | [-43.502, +347.002] | no |
| fixed_negative | final return | +0.103 ± 0.319 | [-0.068, +0.303] | no |
| fixed_negative | recovery_time | -57.900 ± 1393.018 | [-898.518, +800.902] | no |
| adaptive_beta | AUC₀₋₂ₖ | -30.900 ± 155.006 | [-123.903, +57.207] | no |
| adaptive_beta | AUC₀₋₁₀ₖ | +162.800 ± 135.035 | [+87.797, +244.802] | **yes** |
| adaptive_beta | final return | -0.023 ± 0.339 | [-0.219, +0.177] | no |
| adaptive_beta | recovery_time | +261.600 ± 998.485 | [-174.110, +919.005] | no |
| adaptive_sign_only | AUC₀₋₂ₖ | +2.400 ± 122.377 | [-73.500, +70.300] | no |
| adaptive_sign_only | AUC₀₋₁₀ₖ | +52.200 ± 166.444 | [-46.000, +149.602] | no |
| adaptive_sign_only | final return | -0.048 ± 0.238 | [-0.189, +0.089] | no |
| adaptive_sign_only | recovery_time | -387.900 ± 1602.204 | [-1306.105, +521.112] | no |

### HypTest

| method | metric | mean ± std | 95% CI | CI excl. 0 |
|---|---|---|---|---|
| fixed_positive | AUC₀₋₂ₖ | -566.700 ± 179.793 | [-663.903, -458.000] | **yes** |
| fixed_positive | AUC₀₋₁₀ₖ | -11894.200 ± 957.347 | [-12458.100, -11348.595] | **yes** |
| fixed_positive | final return | -1.426 ± 0.479 | [-1.694, -1.135] | **yes** |
| fixed_positive | recovery_time | -31.900 ± 93.247 | [-91.600, -0.600] | **yes** |
| fixed_negative | AUC₀₋₂ₖ | +72.700 ± 142.916 | [-9.602, +155.405] | no |
| fixed_negative | AUC₀₋₁₀ₖ | +1071.500 ± 862.344 | [+527.787, +1522.617] | **yes** |
| fixed_negative | final return | +0.367 ± 0.521 | [+0.075, +0.684] | **yes** |
| fixed_negative | recovery_time | +236.300 ± 477.858 | [-29.900, +541.400] | no |
| adaptive_beta | AUC₀₋₂ₖ | +8.900 ± 175.357 | [-89.502, +115.100] | no |
| adaptive_beta | AUC₀₋₁₀ₖ | +696.600 ± 811.710 | [+227.788, +1169.210] | **yes** |
| adaptive_beta | final return | +0.325 ± 0.452 | [+0.052, +0.581] | **yes** |
| adaptive_beta | recovery_time | +47.400 ± 279.602 | [-91.200, +237.200] | no |
| adaptive_sign_only | AUC₀₋₂ₖ | +29.000 ± 165.171 | [-63.602, +126.102] | no |
| adaptive_sign_only | AUC₀₋₁₀ₖ | +828.800 ± 620.179 | [+448.345, +1170.400] | **yes** |
| adaptive_sign_only | final return | +0.178 ± 0.453 | [-0.079, +0.458] | no |
| adaptive_sign_only | recovery_time | +91.900 ± 291.379 | [-62.000, +280.400] | no |

## 4. adaptive_beta vs fixed_negative (paired)

Tests the spec §17 'fixed-β dominates' criterion. Diffs = adaptive_beta − fixed_negative.

| adversary | metric | mean ± std | 95% CI | CI excl. 0 |
|---|---|---|---|---|
| FM-RM | AUC₀₋₂ₖ | -13.900 ± 149.508 | [-108.605, +64.400] | no |
| FM-RM | AUC₀₋₁₀ₖ | +9.900 ± 333.739 | [-186.400, +206.100] | no |
| FM-RM | final return | -0.126 ± 0.216 | [-0.260, -0.008] | **yes** |
| FM-RM | recovery_time | +319.500 ± 803.369 | [-21.500, +852.505] | no |
| HypTest | AUC₀₋₂ₖ | -63.800 ± 190.739 | [-175.403, +46.900] | no |
| HypTest | AUC₀₋₁₀ₖ | -374.900 ± 566.251 | [-708.200, -45.085] | **yes** |
| HypTest | final return | -0.042 ± 0.641 | [-0.433, +0.330] | no |
| HypTest | recovery_time | -188.900 ± 561.085 | [-537.200, +118.200] | no |

## 5. Strategic-Metric Table (per cell, mean over 10 seeds)

Source: `results/adaptive_beta/strategic/tables/main_strategic_metrics.{csv,tex}`

| game | adversary | method | align | d_eff | opp_ent | pol_TV | shifts | rejections | search | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| shapley | FM-RM | adaptive_beta | 0.890 | 0.375 | 0.923 | 0.399 | 9704 | -- | -- | -- |
| shapley | FM-RM | adaptive_sign_only | 0.905 | 0.468 | 0.924 | 0.398 | 9698 | -- | -- | -- |
| shapley | FM-RM | fixed_negative | 0.906 | 0.469 | 0.924 | 0.398 | 9691 | -- | -- | -- |
| shapley | FM-RM | fixed_positive | 0.054 | 1.769 | 0.756 | 0.418 | 9084 | -- | -- | -- |
| shapley | FM-RM | vanilla | 0.000 | 0.950 | 0.924 | 0.398 | 9692 | -- | -- | -- |
| shapley | HypTest | adaptive_beta | 0.862 | 0.480 | 0.859 | 0.240 | 7843 | 1996 | 6899 | -- |
| shapley | HypTest | adaptive_sign_only | 0.895 | 0.549 | 0.857 | 0.240 | 7830 | 1997 | 6951 | -- |
| shapley | HypTest | fixed_negative | 0.896 | 0.550 | 0.856 | 0.240 | 7822 | 1996 | 6880 | -- |
| shapley | HypTest | fixed_positive | 0.053 | 1.749 | 0.858 | 0.239 | 7844 | 1994 | 7003 | -- |
| shapley | HypTest | vanilla | 0.000 | 0.950 | 0.857 | 0.240 | 7824 | 1995 | 6928 | -- |

## 6. Generated Figures

- learning_curves_main: `/Users/liq/Documents/Claude/Projects/LSE_RL/results/adaptive_beta/strategic/figures/main/learning_curves_main.pdf`
- auc_paired_diff_main: `/Users/liq/Documents/Claude/Projects/LSE_RL/results/adaptive_beta/strategic/figures/main/auc_paired_diff_main.pdf`
- recovery_time_main: `/Users/liq/Documents/Claude/Projects/LSE_RL/results/adaptive_beta/strategic/figures/main/recovery_time_main.pdf`
- event_aligned_main: `/Users/liq/Documents/Claude/Projects/LSE_RL/results/adaptive_beta/strategic/figures/main/event_aligned_main.pdf`
- event_aligned_spec10: `/Users/liq/Documents/Claude/Projects/LSE_RL/results/adaptive_beta/strategic/figures/event_aligned_return.pdf`
- event_aligned_spec10: `/Users/liq/Documents/Claude/Projects/LSE_RL/results/adaptive_beta/strategic/figures/event_aligned_beta.pdf`
- event_aligned_spec10: `/Users/liq/Documents/Claude/Projects/LSE_RL/results/adaptive_beta/strategic/figures/event_aligned_effective_discount.pdf`
- event_aligned_spec10: `/Users/liq/Documents/Claude/Projects/LSE_RL/results/adaptive_beta/strategic/figures/opponent_entropy.pdf`
- mechanism_main: `/Users/liq/Documents/Claude/Projects/LSE_RL/results/adaptive_beta/strategic/figures/main/mechanism_main.pdf`
- beta_trajectory_main: `/Users/liq/Documents/Claude/Projects/LSE_RL/results/adaptive_beta/strategic/figures/main/beta_trajectory_main.pdf`

## 7. §22 Six-Question Checklist

**1. Which strategic settings produced adaptive-β gains? (cite the paired-bootstrap numbers)**

- **FM-RM**: AUC₀₋₂ₖ Δ = -30.900 [-123.903, +57.207]; AUC_full Δ = +162.800 [+87.797, +244.802] *; final Δ = -0.023 [-0.219, +0.177]; recovery Δ = +261.600 [-174.110, +919.005]
- **HypTest**: AUC₀₋₂ₖ Δ = +8.900 [-89.502, +115.100]; AUC_full Δ = +696.600 [+227.788, +1169.210] *; final Δ = +0.325 [+0.052, +0.581] *; recovery Δ = +47.400 [-91.200, +237.200]

Star (*) marks CIs that exclude zero. With 2 settings on 1 game, this is the data we have.

**2. Were gains sample-efficiency, final-return, or recovery gains?**

- **FM-RM**: cumulative AUC
- **HypTest**: cumulative AUC, final-return

**3. Did mechanism metrics support the explanation?**

- **FM-RM** (adaptive_beta): align=0.890, d_eff=0.375, γ=0.95. Mechanism is supportive (align > 0.5 AND d_eff < γ).
- **HypTest** (adaptive_beta): align=0.862, d_eff=0.480, γ=0.95. Mechanism is supportive (align > 0.5 AND d_eff < γ).

**4. Did any fixed β dominate?**

- **FM-RM**: adaptive_beta − fixed_negative AUC_full = +9.900 [-186.400, +206.100] → fixed_negative tied with adaptive_beta (CI overlaps 0). fixed_+ vs vanilla = -32438.300 [-33622.918, -31434.880]; fixed_− vs vanilla = +152.900 [-43.502, +347.002].
- **HypTest**: adaptive_beta − fixed_negative AUC_full = -374.900 [-708.200, -45.085] → **fixed_negative dominates adaptive_beta** (CI excludes 0). fixed_+ vs vanilla = -11894.200 [-12458.100, -11348.595]; fixed_− vs vanilla = +1071.500 [+527.787, +1522.617].

**5. Did any adversary expose a failure mode?**

- **fixed_positive on shapley (both adv)**: final_return collapses to +7.42 vs vanilla +10.43. Pumping continuation up on a strict cycling game amplifies wrong-sign credit assignment. This reproduces the qualitative RPS-style failure mode of fixed_+ on strategic-cycle adversaries.
- adaptive_beta showed **zero divergence** events across all 20 candidate runs — clip is doing its job.

**6. Should the paper be updated, appendix-only, or unchanged?**

**NO UPDATE**. Reasoning:
- fixed_negative ≥ adaptive_beta on auc_full in: ['hypothesis_testing']


## 8. Open Follow-ups

- **Strategic RPS regression** discovered in Stage B2-Dev: across all three endogenous adversaries, adaptive_beta UNDER-PERFORMS vanilla on auc_return (mean Δ AUC = -744, -415, -703 for FM-BR, FM-RM, HypTest respectively at n=3 seeds). The original Phase VII RPS gain may be **adversary-specific to scripted phase opponents**, not a property of adaptive_beta as an endogenous-learning controller. This is a follow-up note (NOT a Stage B2-Main result), and warrants either rerunning the Phase VII RPS claim with endogenous adversaries at higher seed budget, or adding a hedge to the paper narrative explicitly scoping the RPS claim to scripted-phase opponents.
- **Single-game scope.** Stage B2-Main covers only `shapley` × {FM-RM, HypTest}. Two settings on a single game cannot satisfy the spec §17 requirement for ≥2 strategic-game contexts; the verdict reflects this scope limit.
- **fixed_negative-on-Shapley story.** On both adversaries, a static negative β looks competitive with adaptive_beta. This is consistent with Shapley being a cycling game where pessimistic continuation reduces policy chatter near support shifts; it does NOT mean β-control is useless broadly, but it does mean the adaptive controller adds little on this game family beyond what a one-line constant choice gives you.

## 9. Methodological Notes

- Paired-bootstrap percentile CIs (n=10 seeds; BCa not used to avoid small-sample bias in acceleration estimation). 10,000 resamples; fixed seed `0xB2DEF` (mirrors `aggregate.py`).
- `recovery_time` is the first episode at which the SMOOTH_WIN-rolling mean reaches 80% of the asymptotic mean (mean of return[-500:]). On Shapley × {FM-RM, HT}, the `support_shift` flag fires on a majority of episodes (FM-RM ≈ 97%, HT ≈ 78%), so the spec wording 'following a support_shift event' is satisfied by essentially every episode and the metric reduces to the global learning-speed benchmark above. NaN if the threshold is never reached.
- `auc_first_2k` is the spec §15 sample-efficiency primary endpoint; `auc_full` is the secondary cumulative endpoint; `final_return` is the endpoint quoted by the runner top-line table.
