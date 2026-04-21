#!/usr/bin/env bash
# scripts/rerun_phase2_phase3.sh
# Full rerun of Phase II and Phase III experiments after audit fixes.
# Runs sequentially; expected wall time ~3.5 hours.
# Usage: bash scripts/rerun_phase2_phase3.sh 2>&1 | tee logs/rerun_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=".venv/bin/python"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

echo "=== Phase II + III rerun ==="
echo "Start: $(date)"
echo ""

# ---------------------------------------------------------------------------
# Phase II: main DP runs
# ---------------------------------------------------------------------------
echo "--- [1/7] Phase II DP (130 runs) ---"
$PYTHON experiments/weighted_lse_dp/runners/run_phase2_dp.py
echo "Done: $(date)"
echo ""

# ---------------------------------------------------------------------------
# Phase II: main RL runs
# ---------------------------------------------------------------------------
echo "--- [2/7] Phase II RL (64 runs) ---"
$PYTHON experiments/weighted_lse_dp/runners/run_phase2_rl.py
echo "Done: $(date)"
echo ""

# ---------------------------------------------------------------------------
# Phase II: re-aggregate calibration JSONs from fresh DP output
# ---------------------------------------------------------------------------
echo "--- [3/7] Aggregate Phase II calibration stats ---"
$PYTHON experiments/weighted_lse_dp/runners/aggregate_phase2.py
echo "Done: $(date)"
echo ""

# ---------------------------------------------------------------------------
# Phase III: rebuild schedules from corrected calibration JSONs
# ---------------------------------------------------------------------------
echo "--- [4/7] Rebuild Phase III schedules ---"
$PYTHON experiments/weighted_lse_dp/calibration/build_schedule_from_phase12.py \
    --suite-config experiments/weighted_lse_dp/configs/phase3/paper_suite.json
echo "Done: $(date)"
echo ""

# ---------------------------------------------------------------------------
# Phase III: main + all ablation suites (DP + RL)
# ---------------------------------------------------------------------------
PHASE3_CONFIGS=(
    "experiments/weighted_lse_dp/configs/phase3/paper_suite.json"
    "experiments/weighted_lse_dp/configs/phase3/ablation_alpha_0.00.json"
    "experiments/weighted_lse_dp/configs/phase3/ablation_alpha_0.02.json"
    "experiments/weighted_lse_dp/configs/phase3/ablation_alpha_0.05.json"
    "experiments/weighted_lse_dp/configs/phase3/ablation_alpha_0.10.json"
    "experiments/weighted_lse_dp/configs/phase3/ablation_alpha_0.20.json"
    "experiments/weighted_lse_dp/configs/phase3/ablation_beta_zero.json"
    "experiments/weighted_lse_dp/configs/phase3/ablation_beta_constant_small.json"
    "experiments/weighted_lse_dp/configs/phase3/ablation_beta_constant_large.json"
    "experiments/weighted_lse_dp/configs/phase3/ablation_beta_raw_unclipped.json"
)

echo "--- [5/7] Phase III DP (10 configs × 130 runs) ---"
for cfg in "${PHASE3_CONFIGS[@]}"; do
    suite=$(basename "$cfg" .json)
    echo "  Running DP: $suite"
    $PYTHON experiments/weighted_lse_dp/runners/run_phase3_dp.py --config "$cfg"
done
echo "Done: $(date)"
echo ""

echo "--- [6/7] Phase III RL (10 configs × 64 runs) ---"
for cfg in "${PHASE3_CONFIGS[@]}"; do
    suite=$(basename "$cfg" .json)
    echo "  Running RL: $suite"
    $PYTHON experiments/weighted_lse_dp/runners/run_phase3_rl.py --config "$cfg"
done
echo "Done: $(date)"
echo ""

# ---------------------------------------------------------------------------
# Phase II: hyperparameter ablation (1248 runs — longest step)
# ---------------------------------------------------------------------------
echo "--- [7/7] Phase II ablation (1248 runs) ---"
$PYTHON experiments/weighted_lse_dp/runners/run_phase2_ablation.py
echo "Done: $(date)"
echo ""

echo "=== All done: $(date) ==="
