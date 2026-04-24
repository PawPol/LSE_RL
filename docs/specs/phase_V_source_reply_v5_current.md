Use this as the final decision and proceed without more branching questions.

1. **Adopt option 2 now, with a bounded option-3 follow-up.**
   - For **positive families**, the clip gate is no longer “must bind in [0.05, 0.80]”.
   - A positive family may promote either because clipping binds **or** because clipping is provably inactive since the raw schedule is already safe on the certified domain.
   - Log these two cases separately:
     - `promotion_mode = binding_clip`
     - `promotion_mode = safe_active_no_distortion`

2. **Promote Family A immediately.**
   Family A already shows the right constructive effect: near-tie classical state, reachable policy difference, nontrivial `mass_delta_d`, nontrivial value gap. Zero clip fraction does **not** disqualify it.

3. **Keep Family C as the dedicated safety / clipping family.**
   Family C must keep the strict requirement that clipping binds and that raw vs safe differ materially. This is where you show instability of raw and stability of safe.

4. **Do one bounded refinement pass for Family B focused on value translation, not activation.**
   Refine only over a small targeted grid: catastrophe severity, warning depth / timing, matched-concentration variants, gamma subset, asymmetry, and prevention-cost tie tuning.

5. **If Family B still fails after that pass, add Family D immediately** instead of widening B indefinitely.
   - Family D = early-warning preventive-intervention family.
   - Practical interpretation: predictive maintenance / early medical intervention / risk-aware shutdown.
   - This should be the second positive family if B remains weak.

6. **Choose NeurIPS 2026 Theory as the contribution type.**
   The paper is theory-first; the experiments are there to study formalized insights and constructive positive cases, not to be a broad benchmark paper.

7. **Run RL only on strong exact-planning tasks.**
   RL promotion should require at least one of:
   - start-state action flip, or
   - `policy_disagreement >= 0.10`,
   plus a clearly nontrivial value gap.
   Keep RL to at most 2 positive tasks plus 1 optional safety task.

8. **Main paper structure:**
   - Positive family 1: Family A
   - Positive family 2: Family B if refined successfully, else Family D
   - Safety family: Family C
   - Phase I–IV stay appendix-only sanity material

9. **Do not spend time forcing Family A to have active clipping.**
   That is not necessary and will likely make the mechanism story worse.

10. **Next actions in order:**
    - patch promotion logic,
    - re-run shortlist,
    - promote A and C,
    - refine B once,
    - if needed add D,
    - run planning diagnostics,
    - then RL pilot.

That is the final direction.
