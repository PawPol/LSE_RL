#!/usr/bin/env bash
# run_phase2_paper_suite.sh
# Overnight orchestrator for the full Phase II paper suite.
# Runs RL and DP in parallel, then aggregates and generates figures.
#
# Usage:
#   bash scripts/run_phase2_paper_suite.sh
#
# Logs:
#   logs/phase2_rl.log
#   logs/phase2_dp.log
#   logs/phase2_aggregate.log
#   logs/phase2_figures.log
#   logs/phase2_orchestrator.log   <- this script's own output

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

ORCH_LOG="$LOG_DIR/phase2_orchestrator.log"
RL_LOG="$LOG_DIR/phase2_rl.log"
DP_LOG="$LOG_DIR/phase2_dp.log"
AGG_LOG="$LOG_DIR/phase2_aggregate.log"
FIG_LOG="$LOG_DIR/phase2_figures.log"

log() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] $*" | tee -a "$ORCH_LOG"
}

log "===== Phase II paper suite started ====="
log "Repo: $REPO_ROOT"
log "RL log:         $RL_LOG"
log "DP log:         $DP_LOG"
log "Aggregate log:  $AGG_LOG"
log "Figures log:    $FIG_LOG"

# ---------------------------------------------------------------------------
# Step 1 — RL and DP in parallel
# ---------------------------------------------------------------------------
log "Launching RL runner (64 runs, all tasks × algorithms × seeds) ..."
python3 experiments/weighted_lse_dp/runners/run_phase2_rl.py \
    --task all \
    --config experiments/weighted_lse_dp/configs/phase2/paper_suite.json \
    --out-root results/weighted_lse_dp \
    > "$RL_LOG" 2>&1 &
RL_PID=$!
log "RL runner PID: $RL_PID"

log "Launching DP runner (130 runs, all tasks × algorithms × seeds) ..."
python3 experiments/weighted_lse_dp/runners/run_phase2_dp.py \
    --task all \
    --config experiments/weighted_lse_dp/configs/phase2/paper_suite.json \
    --out-root results/weighted_lse_dp \
    > "$DP_LOG" 2>&1 &
DP_PID=$!
log "DP runner PID: $DP_PID"

# ---------------------------------------------------------------------------
# Step 2 — Wait for both
# ---------------------------------------------------------------------------
log "Waiting for RL runner (PID $RL_PID) ..."
RL_EXIT=0
wait $RL_PID || RL_EXIT=$?
if [ "$RL_EXIT" -eq 0 ]; then
    log "RL runner finished OK."
else
    log "WARNING: RL runner exited with code $RL_EXIT. Check $RL_LOG."
fi

log "Waiting for DP runner (PID $DP_PID) ..."
DP_EXIT=0
wait $DP_PID || DP_EXIT=$?
if [ "$DP_EXIT" -eq 0 ]; then
    log "DP runner finished OK."
else
    log "WARNING: DP runner exited with code $DP_EXIT. Check $DP_LOG."
fi

# Abort pipeline if both runners failed completely.
if [ "$RL_EXIT" -ne 0 ] && [ "$DP_EXIT" -ne 0 ]; then
    log "FATAL: Both RL and DP runners failed. Aborting aggregation."
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 3 — Aggregate
# ---------------------------------------------------------------------------
log "Running aggregation ..."
AGG_EXIT=0
python3 experiments/weighted_lse_dp/runners/aggregate_phase2.py \
    --out-root results/weighted_lse_dp \
    > "$AGG_LOG" 2>&1 || AGG_EXIT=$?
if [ "$AGG_EXIT" -eq 0 ]; then
    log "Aggregation finished OK."
else
    log "WARNING: Aggregation exited with code $AGG_EXIT. Check $AGG_LOG."
fi

# ---------------------------------------------------------------------------
# Step 4 — Figures
# ---------------------------------------------------------------------------
log "Generating figures ..."
FIG_EXIT=0
python3 experiments/weighted_lse_dp/analysis/make_phase2_figures.py \
    --out-root results/weighted_lse_dp \
    > "$FIG_LOG" 2>&1 || FIG_EXIT=$?
if [ "$FIG_EXIT" -eq 0 ]; then
    log "Figures finished OK."
else
    log "WARNING: Figure generation exited with code $FIG_EXIT. Check $FIG_LOG."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log ""
log "===== Phase II paper suite complete ====="
log "  RL runner exit:   $RL_EXIT"
log "  DP runner exit:   $DP_EXIT"
log "  Aggregate exit:   $AGG_EXIT"
log "  Figures exit:     $FIG_EXIT"
log ""
log "Raw results:   results/weighted_lse_dp/phase2/raw/"
log "Aggregated:    results/weighted_lse_dp/phase2/aggregated/"
log "Calibration:   results/weighted_lse_dp/phase2/calibration/"
log "Figures:       results/weighted_lse_dp/processed/phase2/figures/"
log ""

OVERALL_EXIT=$(( RL_EXIT + DP_EXIT + AGG_EXIT + FIG_EXIT ))
exit $OVERALL_EXIT
