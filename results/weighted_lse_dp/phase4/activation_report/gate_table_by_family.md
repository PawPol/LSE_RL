# Phase IV-A Activation Gate — Per-Family Diagnostics

- Seed: 42
- Gate condition: `mean_abs_u >= 0.005` AND `frac(|u| >= 5e-3) >= 10%`
- Mainline: T = 20, n_ep = 1000, tau_n = 200
- Appendix: T in [5, 10], n_ep = 200, tau_n = 200
- One row per (family, T, suite). Best-by-mean_abs_u within each group.

| family | T | n_ep | tau_n | mean_abs_u | frac_ge5e3 | c_t_median | u_tr_cap_median | U_safe_median | gate_pass | suite_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chain_catastrophe | 20 | 1000 | 200 | 0.000000 | 0.000 | 0.0000 | 0.000000 | 0.011273 | no | mainline |
| chain_jackpot | 20 | 1000 | 200 | 0.002454 | 0.300 | 0.0000 | 0.000000 | 0.006647 | no | mainline |
| chain_sparse_credit | 20 | 1000 | 200 | 0.005175 | 0.450 | 0.1854 | 0.003920 | 0.023982 | YES | mainline |
| grid_hazard | 20 | 1000 | 200 | 0.003764 | 0.350 | 0.0078 | 0.001392 | 0.025755 | no | mainline |
| regime_shift | 20 | 1000 | 200 | 0.000000 | 0.000 | 0.0000 | 0.000000 | 0.000225 | no | mainline |
| chain_catastrophe | 5 | 200 | 200 | 0.000000 | 0.000 | 0.0000 | 0.000000 | 0.000225 | no | appendix_sanity |
| chain_catastrophe | 10 | 200 | 200 | 0.000000 | 0.000 | 0.0000 | 0.000000 | 0.000225 | no | appendix_sanity |
| chain_jackpot | 5 | 200 | 200 | 0.000000 | 0.000 | 0.0000 | 0.000000 | 0.000225 | no | appendix_sanity |
| chain_jackpot | 10 | 200 | 200 | 0.000000 | 0.000 | 0.0000 | 0.000000 | 0.000620 | no | appendix_sanity |
| chain_sparse_credit | 5 | 200 | 200 | 0.012587 | 1.000 | 0.4257 | 0.012774 | 0.026211 | YES | appendix_sanity |
| chain_sparse_credit | 10 | 200 | 200 | 0.010672 | 1.000 | 0.3446 | 0.010695 | 0.027875 | YES | appendix_sanity |
| grid_hazard | 5 | 200 | 200 | 0.000000 | 0.000 | 0.0000 | 0.000000 | 0.000225 | no | appendix_sanity |
| grid_hazard | 10 | 200 | 200 | 0.002558 | 0.300 | 0.0000 | 0.000000 | 0.010298 | no | appendix_sanity |
| regime_shift | 5 | 200 | 200 | 0.000000 | 0.000 | 0.0000 | 0.000000 | 0.000225 | no | appendix_sanity |
| regime_shift | 10 | 200 | 200 | 0.000000 | 0.000 | 0.0000 | 0.000000 | 0.000225 | no | appendix_sanity |
