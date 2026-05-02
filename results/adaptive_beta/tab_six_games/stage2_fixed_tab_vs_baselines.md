# M7.1 Stage 2 — Fixed TAB vs Q-learning baselines (4 cells × 4 γ × 10 seeds)

- **Created**: 2026-05-02
- **Branch**: `phase-VIII-tab-six-games-2026-04-30`
- **Spec authority**: `docs/specs/phase_VIII_tab_six_games.md` §10.3 (Stage 2)
- **Verdict**: **PASS — M7.1 → M8 acceptance criteria met.** ≥ 1 G_+
  candidate (AC-Trap γ=0.60), ≥ 1 G_− candidate (multiple cells across
  γ). Honesty-rule disclosures of baseline wins are below.
- **Headline scientific finding**: **simple non-stationary baselines
  (`tuned_epsilon_greedy`, `restart_Q_learning`) match or outperform
  TAB on three of the four headline cells**; TAB's distinctive value
  is concentrated on the deep-delayed-credit chain (DC-Long50), where
  no baseline beats vanilla but `fixed_beta_-2.0` does (Δ = +795 to
  +2 837 across γ).

## 1. Dispatch summary

| pass | runs | wall | status |
|---|---:|---:|---|
| Tier II TAB re-dispatch (10 seeds) | 3 360 | ~125 min | 3 360 / 3 360 ✓ |
| Stage 2 baselines v1 (defective config) | 480 | ~12 min | 480 / 480 ✓ but 2 of 3 baselines bit-identical to vanilla; superseded |
| Stage 2 baselines v2 (fixed config) | 480 | ~14 min | 480 / 480 ✓ canonical |

Total compute on this milestone: **3 840 main-pass runs + 480 v1 runs
discarded as superseded**.

## 2. Pairing/config verification (per user required-before-closing #1)

Sample run from each pass cross-checked at `(rules_of_road,
RR-StationaryConvention, gamma=0.95, seed=2)`:

| key | TAB (m7_1_tier2_tab_redispatch_10seeds) | Baselines (m7_1_stage2_baselines_v2_headline) | match |
|---|---|---|---|
| gamma | 0.95 | 0.95 | ✓ |
| q_init | 0.0 | 0.0 | ✓ |
| learning_rate | 0.1 | 0.1 | ✓ |
| episodes | 10 000 | 10 000 | ✓ |
| seed | 2 | 2 | ✓ |
| game | rules_of_road | rules_of_road | ✓ |
| subcase | RR-StationaryConvention | RR-StationaryConvention | ✓ |
| adversary | stationary_mixed | stationary_mixed | ✓ |
| epsilon_schedule (runner-built) | start=1.0, end=0.05, decay=5000 | identical | ✓ |
| seed list | 0–9 | 0–9 | ✓ |

Pairing contract holds: every (cell, γ, seed) row of the M7.1 long CSV
has TAB ↔ baseline pairing intact.

**Note (per user required-before-closing #4)**: although the user
authorised re-dispatch under the verifier-mismatch escape clause, the
TAB and baseline runs in this milestone are NOT reused V10 data —
both passes are fresh dispatches authored 2026-05-02. The earlier 5-seed
V10 Tier II remains on disk as a historical artifact and is NOT joined
into the M7.1 long CSV; the canonical paired comparison uses only the
3 360 + 480 = 3 840 fresh runs.

## 3. v1 → v2 baseline-config supersession (the discovery story)

The v1 dispatch (`stage2_baselines_headline.yaml`, 480 runs) produced
`sliding_window_Q_learning` and `tuned_epsilon_greedy_Q_learning`
results **bit-identical to vanilla** for every (cell, γ, seed). Root
causes:

1. **Tuned ε override.** The Stage 2 runner was passing the YAML's
   ε-schedule (vanilla: start=1.0, end=0.05, decay=5000) to the
   `TunedEpsilonGreedyQLearningAgent` constructor as
   `epsilon_schedule=eps_fn`. The class accepts a `None` default and
   builds its tuned schedule (start=1.0, end=0.01, decay=2000) only
   when `epsilon_schedule is None`. The runner's override neutralised
   the entire "tuned" mechanism. **Fixed** in
   `_build_baseline_agent`: `epsilon_schedule=None` for the tuned
   branch unless the user explicitly sets one via `method_kwargs`.

2. **Sliding-window default too large.** `window_size=10000` (class
   default) vs the v2 dispatch's 10 000-episode × ~20-step horizon =
   200 000 transitions. The buffer never fills past its maxlen long
   enough to evict states, so no state-reset ever fired and the
   update rule reduced to vanilla. **Fixed** via per-method config
   `method_kwargs_per_method.sliding_window_Q_learning.window_size:
   2000` (≈ 100 episodes for matrix games at horizon 20, ≈ 40
   episodes for DC-Long50). Eviction now fires routinely.

3. **(Documentation gap, not a runtime bug):** the runner's
   `ep_epsilon` array records the runner's vanilla ε schedule even
   for `tuned_epsilon_greedy_Q_learning`, where the agent internally
   uses the class tuned default. AUC and `bellman_residual` correctly
   reflect the agent's actual ε behaviour (the class is queried for
   action selection); only the per-episode logging field is mis-named.
   Recommend a future patch to query `agent.current_epsilon()` for
   `ep_epsilon`. **Does not affect any M7.1 conclusion.**

The v2 patch passed a single-seed smoke (4 cells × 4 γ × 3 methods =
48 runs at seed 0); restart, sliding-window, and tuned-ε produced
distinguishably different AUCs from each other and from vanilla. The
full 480-run v2 main pass then completed clean.

The v1 raw artifacts are retained at
`raw/VIII/m7_1_stage2_baselines_headline/` for traceability and are
explicitly excluded from the canonical aggregate
(`processed/m7_1_long.csv` rows are stamped with stage names).

## 4. Paired-comparison results (per user required-before-closing #2)

Methodology: paired-bootstrap CI (B = 20 000) on `headline_AUC`
difference vs vanilla, paired by seed at the (cell, γ) level. Headline
metric per spec: cumulative-return AUC for matrix games;
`-log(bellman_residual)` AUC (v5b advance-only metric) for DC-Long50.

Significance flag:
- ✓ : CI₉₅ strictly above 0 (arm beats vanilla)
- ✗ : CI₉₅ strictly below 0 (arm loses to vanilla)
- 0 : CI straddles 0 (no difference detectable)

### 4.1 AC-Trap

| γ | method | source | Δ | CI₉₅ | sig |
|---|---|---|---:|---|---|
| 0.60 | best_fixed_positive_TAB | `fixed_beta_+0.1` | **+131.7** | [+87.8, +174.2] | **✓** |
| 0.60 | best_fixed_negative_TAB | `fixed_beta_-0.1` | -52.4 | [-97.0, -10.2] | ✗ |
| 0.60 | best_fixed_beta_grid | `fixed_beta_+0.1` | **+131.7** | [+87.8, +174.2] | **✓** |
| 0.60 | restart_Q_learning | `restart_Q_learning` | **+250 596.6** | [+248 254, +252 827] | **✓** |
| 0.60 | sliding_window_Q_learning | `sliding_window_Q_learning` | -3 068.6 | [-3 594, -2 518] | ✗ |
| 0.60 | tuned_epsilon_greedy_Q_learning | `tuned_epsilon_greedy_Q_learning` | **+44 194.1** | [+43 878, +44 512] | **✓** |
| 0.80 | best_fixed_positive_TAB | `fixed_beta_+0.05` | -43.9 | [-147.5, +49.4] | 0 |
| 0.80 | best_fixed_negative_TAB | `fixed_beta_-0.05` | -44.5 | [-71.1, -21.1] | ✗ |
| 0.80 | best_fixed_beta_grid | `vanilla` | 0 | (none) | 0 |
| 0.80 | restart_Q_learning | `restart_Q_learning` | **+249 342.5** | [+246 931, +251 898] | **✓** |
| 0.80 | sliding_window_Q_learning | — | -3 289.9 | [-3 925, -2 651] | ✗ |
| 0.80 | tuned_epsilon_greedy_Q_learning | — | **+44 215.7** | [+43 880, +44 555] | **✓** |
| 0.90 | best_fixed_positive_TAB | `fixed_beta_+0.05` | -40 266.3 | [-42 093, -38 726] | ✗ |
| 0.90 | best_fixed_negative_TAB | `fixed_beta_-0.05` | -100.3 | [-157.4, -39.5] | ✗ |
| 0.90 | best_fixed_beta_grid | `vanilla` | 0 | (none) | 0 |
| 0.90 | restart_Q_learning | — | **+238 560.0** | [+213 586, +252 175] | **✓** |
| 0.90 | sliding_window_Q_learning | — | -3 436.4 | [-4 074, -2 811] | ✗ |
| 0.90 | tuned_epsilon_greedy_Q_learning | — | **+44 047.5** | [+43 566, +44 490] | **✓** |
| 0.95 | best_fixed_positive_TAB | `fixed_beta_+0.05` | -50 924.6 | [-51 924, -49 830] | ✗ |
| 0.95 | best_fixed_negative_TAB | `fixed_beta_-0.05` | -80.8 | [-141.8, -27.2] | ✗ |
| 0.95 | best_fixed_beta_grid | `vanilla` | 0 | (none) | 0 |
| 0.95 | restart_Q_learning | — | **+248 462.7** | [+245 970, +250 984] | **✓** |
| 0.95 | sliding_window_Q_learning | — | -3 553.5 | [-4 234, -2 860] | ✗ |
| 0.95 | tuned_epsilon_greedy_Q_learning | — | **+43 485.4** | [+43 089, +43 857] | **✓** |

**AC-Trap takeaways**: TAB at γ = 0.60 narrowly beats vanilla at β=+0.10
(+131.7); at higher γ TAB does not. **Restart Q-learning beats vanilla
by 47 000 % more (Δ ≈ +250 000) at every γ** — restart triggers fire
when the agent gets stuck in the trap, forcing re-exploration.
**Tuned-ε also beats vanilla by +44 000 across all γ** — the lower
ε floor (0.01 vs 0.05) and slower decay (2 000 vs 5 000 episodes)
combine to escape the trap via continued exploration. Sliding-window
(window=2 000) loses ~3 000 — the eviction-driven Q resets
destabilise the policy on a deceptive game.

### 4.2 RR-StationaryConvention

| γ | method | source | Δ | CI₉₅ | sig |
|---|---|---|---:|---|---|
| 0.60 | best_fixed_positive_TAB | `fixed_beta_+0.05` | -20.8 | [-41.8, -6.0] | ✗ |
| 0.60 | best_fixed_negative_TAB | `fixed_beta_-0.5` | **+26.0** | [+8.2, +42.8] | **✓** |
| 0.60 | best_fixed_beta_grid | `fixed_beta_-0.5` | **+26.0** | [+8.2, +42.8] | **✓** |
| 0.60 | restart_Q_learning | — | -2 155.8 | [-2 321, -2 010] | ✗ |
| 0.60 | sliding_window_Q_learning | — | 0 | (none) | 0 |
| 0.60 | tuned_epsilon_greedy_Q_learning | — | **+14 465.6** | [+14 281, +14 642] | **✓** |
| 0.80 | best_fixed_negative_TAB | `fixed_beta_-0.5` | **+124.2** | [+87.2, +164.2] | **✓** |
| 0.80 | tuned_epsilon_greedy_Q_learning | — | **+14 468.0** | [+14 290, +14 640] | **✓** |
| 0.90 | best_fixed_negative_TAB | `fixed_beta_-0.5` | **+206.8** | [+157.8, +260.0] | **✓** |
| 0.90 | tuned_epsilon_greedy_Q_learning | — | **+14 468.0** | [+14 255, +14 673] | **✓** |
| 0.95 | best_fixed_negative_TAB | `fixed_beta_-0.5` | **+320.0** | [+241.2, +401.4] | **✓** |
| 0.95 | tuned_epsilon_greedy_Q_learning | — | **+14 419.0** | [+14 233, +14 602] | **✓** |

(Restart loses to vanilla by ~3 000–5 000 at every γ; sliding-window is
indistinguishable from vanilla.)

**RR takeaways**: TAB-fixed_beta_-0.5 wins vs vanilla at every γ but
the magnitude (+26 to +320) is **two orders smaller than tuned-ε's
gain (~+14 500 at every γ)**. `tuned_epsilon_greedy_Q_learning` is the
strict best method on this cell. Restart fails — its trigger fires
on a stationary task and destroys convergence.

### 4.3 SH-FiniteMemoryRegret

| γ | method | source | Δ | CI₉₅ | sig |
|---|---|---|---:|---|---|
| 0.60 | best_fixed_positive_TAB | `fixed_beta_+0.35` | +84.1 | [-39.2, +219.9] | 0 |
| 0.60 | best_fixed_negative_TAB | `fixed_beta_-0.05` | +38.7 | [-60.8, +145.2] | 0 |
| 0.60 | restart_Q_learning | — | -20 378.6 | [-23 747, -17 040] | ✗ |
| 0.60 | tuned_epsilon_greedy_Q_learning | — | **+11 402.5** | [+11 300, +11 495] | **✓** |
| 0.80 | best_fixed_negative_TAB | `fixed_beta_-0.5` | **+97.3** | [+9.9, +181.8] | **✓** |
| 0.80 | tuned_epsilon_greedy_Q_learning | — | **+11 442.6** | [+11 291, +11 590] | **✓** |
| 0.90 | best_fixed_negative_TAB | `fixed_beta_-0.2` | +99.4 | [-68.5, +264.4] | 0 |
| 0.90 | tuned_epsilon_greedy_Q_learning | — | **+11 434.7** | [+11 255, +11 614] | **✓** |
| 0.95 | best_fixed_negative_TAB | `fixed_beta_-0.5` | +147.8 | [-14.9, +334.6] | 0 |
| 0.95 | tuned_epsilon_greedy_Q_learning | — | **+11 582.9** | [+11 469, +11 711] | **✓** |

**SH-FMR takeaways**: TAB has a single CI-significant win
(`fixed_beta_-0.5` at γ=0.80, Δ=+97.3); at γ ∈ {0.6, 0.9, 0.95} TAB's
best arm has CI straddling 0. `tuned_epsilon_greedy` wins by
~+11 500 at every γ — again, two orders larger than TAB's best win.
Restart loses by ~20 000.

### 4.4 DC-Long50

| γ | method | source | Δ | CI₉₅ | sig |
|---|---|---|---:|---|---|
| 0.60 | best_fixed_negative_TAB | `fixed_beta_-2.0` | **+794.9** | (deterministic) | **✓** |
| 0.60 | restart_Q_learning | — | 0 | (deterministic) | 0 |
| 0.60 | sliding_window_Q_learning | — | 0 | (deterministic) | 0 |
| 0.60 | tuned_epsilon_greedy_Q_learning | — | 0 | (deterministic) | 0 |
| 0.80 | best_fixed_negative_TAB | `fixed_beta_-2.0` | **+1 510.2** | (deterministic) | **✓** |
| 0.90 | best_fixed_negative_TAB | `fixed_beta_-2.0` | **+2 192.4** | (deterministic) | **✓** |
| 0.95 | best_fixed_negative_TAB | `fixed_beta_-2.0` | **+2 837.1** | (deterministic) | **✓** |

(Per spec §10.2 v5b, DC-Long50's deterministic chain dynamics produce
zero across-seed variance; the relative-gap floor is the t11 guard, not
Cohen's d. Identical paired Δ is the expected behaviour, not bootstrap
collapse.)

**DC-Long50 takeaways**: **TAB is the only thing that beats vanilla
on DC-Long50**. All three baselines reduce to vanilla because (a) Q-
learning on this deterministic chain converges to the same Q* under
any sufficient exploration policy; (b) restart's return-history
rolling-mean trigger never fires (returns are stable at 1 per
episode); (c) sliding-window's eviction-driven state-resets never
fire (the small chain state space is fully covered every few
episodes); (d) tuned-ε's modified schedule converges to the same
Q-table as vanilla. **`fixed_beta_-2.0` produces a Δ of +795 at
γ=0.60 rising monotonically to +2 837 at γ=0.95** — TAB's
distinctive value on the deep-delayed-credit task.

## 5. G_+ / G_− classification (per spec §10.4 M8 input)

| cell | G_+ at γ ∈ | G_− at γ ∈ |
|---|---|---|
| AC-Trap | **{0.60}** | ∅ |
| RR-StationaryConvention | ∅ | **{0.60, 0.80, 0.90, 0.95}** |
| SH-FiniteMemoryRegret | ∅ | **{0.80}** |
| DC-Long50 | ∅ | **{0.60, 0.80, 0.90, 0.95}** |

**Acceptance criterion (spec §10.4)**: ≥ 1 G_+ candidate AND ≥ 1 G_−
candidate. **MET**: AC-Trap γ=0.60 (G_+) and 9 G_− cells across the
remaining three subcases. **M7.1 → M8 promotion permitted by spec
§10.4 acceptance gate.**

## 6. Honesty-rule disclosures (per spec §10.3)

The spec mandates "honesty rule: if sliding-window or restart wins in
some cells, report it." Per the tables above:

- **`tuned_epsilon_greedy_Q_learning` strictly beats `best_fixed_*_TAB`
  on AC-Trap, RR-StationaryConvention, and SH-FiniteMemoryRegret at
  every γ**. The gain is two orders larger than TAB's best gain on
  these cells (Δ ≈ 11 000–44 000 vs TAB's Δ ≈ 26–320). This is a
  serious challenge to the headline TAB-as-sufficient-mechanism claim
  — a tuned ε schedule alone matches or beats the operator-side
  intervention on three of four headline cells.

- **`restart_Q_learning` strictly beats `best_fixed_*_TAB` on AC-Trap
  by ~+250 000 AUC** at every γ. Restart's mechanism (return-drop
  triggered Q-reset) is dramatically more effective than TAB on the
  deceptive coordination task. This is consistent with AC-Trap being
  framed by spec §5.4 v7 as a *falsifiability* cell — restart's
  success here is the "expected baseline win" the spec anticipates.

- **`sliding_window_Q_learning`** (window=2 000) does NOT beat vanilla
  on any cell; on AC-Trap it loses ~3 000. The eviction-driven
  state-reset mechanism is destabilising on these task families.
  No baseline-win is reported here.

The narrative implication for the paper:

> TAB's distinctive contribution is on long-horizon delayed-credit
> tasks (DC-Long50). On stationary matrix games, simple non-stationary
> baselines (a tuned ε schedule, periodic restart) match or
> outperform any fixed β regime. The paper headline must reframe the
> claim: TAB is not a universal Bellman-operator improvement but a
> specialised mechanism for the credit-assignment regime where the
> reward signal is deeply delayed.

## 7. Limitations (per user required-before-closing #4)

1. **Strategic-learning agent baselines deferred to M7.2.** Spec §6.3
   patch §3 (2026-05-01) promotes `regret_matching_agent` and
   `smoothed_fictitious_play_agent` from M11 to M7. Their agent
   wrappers are not yet implemented (only the opponent classes
   exist). This is the largest gap in M7.1's baseline coverage; per
   user's M7 directive, deferring to M7.2 is approved.

2. **6-game scope deferred.** Spec §10.3 calls for "all six games +
   delayed_chain × DC-Long50". M7.1 uses the user-narrowed 4 Tier II
   headline cells. The remaining 26 cells are TAB-only at γ=0.95
   (V10 Tier I) and lack baseline coverage. Out of M7.1 scope by
   design; M7.3 candidate.

3. **`ep_epsilon` logging artefact** (§3, item 3). The
   `tuned_epsilon_greedy_Q_learning` runs record the runner's vanilla
   ε schedule in `metrics.npz::epsilon` instead of the agent's
   actual tuned schedule. AUC and bellman_residual are correct (they
   integrate the agent's true behaviour); only the logged ε array is
   wrong. **Does not affect any M7.1 conclusion.** Recommend patching
   the runner to query `agent.current_epsilon()` for the recorded
   value.

4. **DC-Long50 paired CI is point-mass.** All 10 seeds produce
   identical AUC for any (method, γ) pair due to the chain's
   deterministic dynamics under passive adversary. CI is reported
   as "(deterministic)" — paired-bootstrap is not the right tool
   here, but the per-seed identity itself is the proof of effect:
   either the method differs from vanilla (Δ ≠ 0, deterministic) or
   it does not (Δ = 0). Per spec v5b §10.2, the relative-gap floor
   guards H1 here, not Cohen's d. M7.1 inherits the same convention.

5. **No re-use of V10 5-seed Tier II data**. Per user M7 directive
   option (β), the M7.1 paired comparison uses only the fresh 10-seed
   re-dispatch. The earlier V10 Tier II 5-seed runs are retained but
   not joined. No "limitation from reusing V10 data" applies.

## 8. Manifest (reused-data marking, per user required-before-closing #3)

The M7.1 long CSV (`processed/m7_1_long.csv`, 3 840 rows) carries a
`stage` column with one of two values:

- `m7_1_tier2_tab_redispatch_10seeds` (3 360 rows) — TAB methods,
  fresh 10-seed re-dispatch.
- `m7_1_stage2_baselines_v2_headline` (480 rows) — 3 baselines, fresh
  v2 dispatch.

Neither stage joins V10 data. Pre-M7.1 V10 Tier II artifacts at
`raw/VIII/v10_tier2_gamma_beta_headline/` (1 680 runs at 5 seeds)
remain on disk for traceability and are explicitly excluded.

## 9. Acceptance for M7.1 → M7.2 / M8 promotion

- ✓ Paired-seed comparison TAB vs baselines vs vanilla on all 4 cells × 4 γ
- ✓ No silent drops (3 360 + 480 = 3 840 / 3 840 status: completed)
- ✓ Honesty rule: explicit baseline-win cells reported (§6)
- ✓ ≥ 1 G_+ candidate AND ≥ 1 G_− candidate (§5)
- **Pending: user sign-off on M7.1 → M7.2 / M8 transition.**

Recommended next steps (user decision required):
- **(i) M7.2**: implement strategic-learning agent baselines
  (`regret_matching_agent`, `smoothed_fictitious_play_agent`) and
  re-run on the 4 Tier II cells. This is the largest gap in M7's
  baseline coverage.
- **(ii) M8**: spec-mandated sign-specialisation analysis (no new
  runs). Inputs are the §5 G_+ / G_− classification.
- **(iii) Paper-headline reframing**: surface the §6 honesty disclosure
  as a substantive scientific finding (TAB is specialised for
  deep-delayed-credit; not a universal operator improvement).

## Reproduction

```bash
# TAB re-dispatch (3 360 runs, ~2 h)
.venv/bin/python -m experiments.adaptive_beta.tab_six_games.runners.run_phase_VIII_stage1_beta_sweep \
    --config experiments/adaptive_beta/tab_six_games/configs/stage2_tier2_redispatch_10seeds.yaml \
    --output-root results/adaptive_beta/tab_six_games

# Baselines v2 (480 runs, ~14 min)
.venv/bin/python -m experiments.adaptive_beta.tab_six_games.runners.run_phase_VIII_stage2_baselines \
    --config experiments/adaptive_beta/tab_six_games/configs/stage2_baselines_headline_v2.yaml \
    --output-root results/adaptive_beta/tab_six_games

# Aggregate + paired CI (~30 sec)
.venv/bin/python scripts/figures/phase_VIII/m7_aggregate.py
```
