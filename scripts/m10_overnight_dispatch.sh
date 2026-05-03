#!/usr/bin/env bash
# M10 overnight UCB dispatch driver.
#
# Runs Phases A, B, C (4+3 hyperparameter sweeps), D sequentially.
# Logs each phase to /tmp/m10_phase_<X>.log.
#
# Total compute estimate: ~3.5 hours wall.
#
# Phase A: 30 cells × 4 γ × 10 seeds × 2 UCB    = 2400 runs (~50 min)
# Phase B: M9 composite × 4 dwells × 10 × 2 UCB = 80   runs (~30 min)
# Phase C: 4 cells × {4 γ × 5 seeds × 2 × 4 c}  = 640  runs (~15 min)
#         + 4 cells × γ=0.95 × 5 seeds × 2 × 3w = 120  runs (~3  min)
# Phase D: 4 cells × 4 γ × 10 seeds × 2 × 50k   = 320  runs at 5x (~40 min)
# Total: ~3560 runs, ~140 min compute time.
set -euo pipefail
cd /Users/liq/Documents/Claude/Projects/LSE_RL

VENV=.venv/bin/python
ROOT=results/adaptive_beta/tab_six_games
STAGE5=experiments.adaptive_beta.tab_six_games.runners.run_phase_VIII_stage5_adaptive
STAGE4=experiments.adaptive_beta.tab_six_games.runners.run_phase_VIII_stage4_composite
LOGDIR=/tmp

echo "==== Phase A: Canonical UCB benchmark (2400 runs) ===="
$VENV -m $STAGE5 \
    --config experiments/adaptive_beta/tab_six_games/configs/m10_phase_A_ucb_canonical.yaml \
    --output-root $ROOT > $LOGDIR/m10_phase_A.log 2>&1

echo "==== Phase B: UCB on M9 composite (80 runs) ===="
$VENV -m $STAGE4 \
    --config experiments/adaptive_beta/tab_six_games/configs/m10_phase_B_ucb_composite.yaml \
    --output-root $ROOT > $LOGDIR/m10_phase_B.log 2>&1

echo "==== Phase C1: UCB c-sweep (4 sub-configs, 640 runs total) ===="
for c_tag in c0_5 c1_0 c1_5 c2_0; do
    echo "  Phase C1 / $c_tag"
    $VENV -m $STAGE5 \
        --config experiments/adaptive_beta/tab_six_games/configs/m10_phase_C1_${c_tag}.yaml \
        --output-root $ROOT > $LOGDIR/m10_phase_C1_${c_tag}.log 2>&1
done

echo "==== Phase C2: SKIPPED — ContractionUCBBetaSchedule has no warm_start_pulls knob ===="
# Warm-start = len(arm_grid), hard-coded. Varying it would require either
# modifying the schedule class (out of scope) or substituting different
# arm-grid lengths (different mechanism, different paper claim).
# C2 configs are retained on disk for later rework.

echo "==== Phase D: Extended-horizon UCB (50k ep, 320 runs ≈ 1600 std equiv) ===="
$VENV -m $STAGE5 \
    --config experiments/adaptive_beta/tab_six_games/configs/m10_phase_D_ucb_extended_horizon.yaml \
    --output-root $ROOT > $LOGDIR/m10_phase_D.log 2>&1

echo "==== M10 overnight dispatch COMPLETE ===="
date
