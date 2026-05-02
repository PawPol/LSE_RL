#!/usr/bin/env bash
# Regenerate Phase VIII v10 figures + tables from raw metrics.npz.
#
# Usage:
#   bash scripts/figures/phase_VIII/regenerate_v10.sh
#
# Inputs:
#   results/adaptive_beta/tab_six_games/raw/VIII/v10_tier{1,2,3}_*/manifest.jsonl
# Outputs:
#   results/adaptive_beta/tab_six_games/figures/v10/{*.pdf, *.png}
#   results/adaptive_beta/tab_six_games/figures/v10/tables/{*.csv}
set -euo pipefail
cd "$(dirname "$0")/../../.."  # repo root

PY=.venv/bin/python
DIR=scripts/figures/phase_VIII

$PY $DIR/v10_aggregate.py
$PY $DIR/v10_build_dispositions.py
$PY $DIR/v10_fig1_beta_vs_auc.py
$PY $DIR/v10_fig2_gamma_beta_heatmap.py
$PY $DIR/v10_fig3_alignment_collapse.py
$PY $DIR/v10_fig4_divergence_signature.py
$PY $DIR/v10_fig5_dc_long50_residual.py

echo
echo "=== outputs ==="
cd results/adaptive_beta/tab_six_games/figures/v10
shasum -a 256 *.pdf tables/*.csv
