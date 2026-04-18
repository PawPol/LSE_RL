# Phase II R9 — Standard Codex Review

**Base**: main  
**Branch**: phase-II/closing  
**Model**: gpt-5.4  
**Date**: 2026-04-17

---

## Summary

Two data-presentation bugs: adaptation curves are shifted in episode space, and visitation heatmaps omit terminal arrivals. Both produce incorrect published artifacts even when underlying runs are valid.

---

## Findings

### [P2] Align rolling means with the episodes they summarize
**File**: `experiments/weighted_lse_dp/analysis/make_phase2_figures.py:498-501`

When `window > 1` (default is 20), `np.convolve(..., mode="valid")` produces a point for episodes `i..i+window-1`, but `ep_rolling = episodes[:len(rolling)]` plots that point at episode `i` instead of the window end (or center). This shifts the whole adaptation curve left by `window-1` episodes, so the plotted recovery appears earlier than it really is and the `change_at_episode` marker is misaligned in every generated adaptation figure.

### [P2] Include terminal arrivals when aggregating visitation counts
**File**: `experiments/weighted_lse_dp/runners/aggregate_phase2.py:588-597`

This aggregation only bins `transitions["state"]`, i.e. the source state of each step. For the Phase II grid tasks, the goal is reached on the last transition and is typically never used as a source state afterward, so the goal cell is systematically undercounted (often zero) in `visitation_counts`. The resulting heatmaps misreport which cells the learned policy actually visits most often.
