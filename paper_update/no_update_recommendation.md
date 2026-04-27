# Paper-update recommendation: NO UPDATE

**Source:** Phase VII-B Stage B2-Main verdict (`results/adaptive_beta/strategic/stage_B2_main_summary.md`).

**Reasoning:**
- fixed_negative ≥ adaptive_beta on auc_full in: ['hypothesis_testing']

## Headline paired-bootstrap numbers

| adversary | comparison | metric | mean ± CI | CI excl 0 |
|---|---|---|---|---|
| FM-RM | adaptive_beta_vs_vanilla | auc_first_2k | -30.900 [-123.903, +57.207] | no |
| FM-RM | adaptive_beta_vs_fixed_negative | auc_first_2k | -13.900 [-108.605, +64.400] | no |
| FM-RM | adaptive_beta_vs_vanilla | auc_full | +162.800 [+87.797, +244.802] | yes |
| FM-RM | adaptive_beta_vs_fixed_negative | auc_full | +9.900 [-186.400, +206.100] | no |
| FM-RM | adaptive_beta_vs_vanilla | recovery_time | +261.600 [-174.110, +919.005] | no |
| FM-RM | adaptive_beta_vs_fixed_negative | recovery_time | +319.500 [-21.500, +852.505] | no |
| HypTest | adaptive_beta_vs_vanilla | auc_first_2k | +8.900 [-89.502, +115.100] | no |
| HypTest | adaptive_beta_vs_fixed_negative | auc_first_2k | -63.800 [-175.403, +46.900] | no |
| HypTest | adaptive_beta_vs_vanilla | auc_full | +696.600 [+227.788, +1169.210] | yes |
| HypTest | adaptive_beta_vs_fixed_negative | auc_full | -374.900 [-708.200, -45.085] | yes |
| HypTest | adaptive_beta_vs_vanilla | recovery_time | +47.400 [-91.200, +237.200] | no |
| HypTest | adaptive_beta_vs_fixed_negative | recovery_time | -188.900 [-537.200, +118.200] | no |

## What the paper should NOT do

- Do not extend the RPS adaptive_beta claim to Shapley without further data.
- Do not claim adaptive_beta beats fixed-β consistently — it does not on this game family.
