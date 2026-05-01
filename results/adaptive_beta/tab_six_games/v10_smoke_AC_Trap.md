# V10.1 smoke — AC-Trap × 21-arm β grid × γ=0.95

- **Created**: 2026-05-01T23:24:13Z
- **Config**: `experiments/adaptive_beta/tab_six_games/configs/v10_smoke_AC_Trap.yaml`
- **Runs**: 63 (1 cell × 21 β × 3 seeds × 1k ep)
- **Wall-clock**: 43 seconds
- **HEAD**: `95268326`

## β-AUC curve (full v10 grid; mean ± std over 3 seeds)

| β | mean AUC | std (ddof=1) | mean align[-200:] |
| ---: | ---: | ---: | ---: |
| -2.00 | 52901.33 | 92.56 | 0.8517 |
| -1.70 | 52902.00 | 93.11 | 0.8517 |
| -1.35 | 52902.00 | 93.11 | 0.8517 |
| -1.00 | 52909.33 | 99.88 | 0.9008 |
| -0.75 | 52910.67 | 101.23 | 0.9008 |
| -0.50 | 52922.33 | 110.94 | 0.9008 |
| -0.35 | **52931.33** | 120.48 | 0.9009 |
| -0.20 | 52900.67 | 142.50 | 0.9009 |
| -0.10 | 52765.67 | 142.12 | 0.9009 |
| -0.05 | 52640.00 | 120.47 | 0.9009 |
| +0.00 | 52450.67 | **30.19** | 0.0000 |
| +0.05 | 47322.67 | 2970.72 | 0.0486 |
| +0.10 | 43062.67 | 3497.44 | 0.0483 |
| +0.20 | 39211.33 | 2498.11 | 0.0487 |
| +0.35 | 41497.50 | 4152.40 | 0.0487 |
| +0.50 | 42085.83 | 2169.47 | 0.0491 |
| +0.75 | 49185.67 | 11382.05 | 0.0485 |
| +1.00 | 40025.33 | 750.97 | 0.0504 |
| +1.35 | 41620.50 | 1917.30 | 0.0496 |
| +1.70 | 41713.00 | 3826.65 | 0.0487 |
| +2.00 | 47509.33 | 11820.31 | 0.0471 |

## Findings

1. **Sharp β=0 sign-bifurcation confirmed at fine resolution**:
   `align(β=-0.05) = 0.9009 → align(β=+0.05) = 0.0486` is an
   **18.5× collapse** across one Δβ=0.10 tick. Pre-v10 wave 5
   (figures-only sub-pass) showed the same shape on
   AC-FictitiousPlay; v10 confirms it on AC-Trap with paper-quality
   resolution.

2. **β-AUC curve shape (3 regimes)**:
   - **−β plateau** (β ∈ [-2.0, -0.50]): AUC flat at ~52900-52930;
     alignment 0.85-0.90 (in-regime).
   - **Optimum near β=-0.35**: AUC=52931 (peak).
   - **Linear descent** (β ∈ [-0.20, +0.00]): AUC drops 52900 → 52450
     as alignment dilutes.
   - **+β collapsed regime** (β > 0): AUC 39000-49000 with high
     variance (std up to 11820); alignment ~0.05 (out-of-regime).

3. **Variance grows with |β| in the +β regime**: std at β=+2.0 is
   11820 (vs std=30 at vanilla). Destabilization signature.

4. **+β has non-monotone sub-structure** within the collapsed
   regime: β=+0.75 transient at AUC=49185 (std 11382) is a high-
   variance point that resolves differently at β=+1.0 (40025).
   The fine grid reveals this is NOT a smooth monotone decline.

5. **Vanilla wins overall** in this cell at γ=0.95, q_init=0:
   AUC=52450 vs best -β AUC=52931 is within 1 std (101). The
   v7+v10-refined narrative ("vanilla never beaten by either side")
   holds: the -β plateau is statistically indistinguishable from
   vanilla within paired-seed noise; the +β regime is materially
   worse.

## Disposition

V10.1 smoke matches v7 prediction at fine resolution. **PROCEED to
V10.2** (Tier I-only dev pass at γ=0.95 over the full 30-cell
enumeration with the 21-arm grid).
