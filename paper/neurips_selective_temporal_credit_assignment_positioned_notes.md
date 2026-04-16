# Literature-positioning revision

This revision tightens the literature discussion in the NeurIPS draft.

Main changes:

- Rewrote the introduction's related-work paragraph so that each adjacent method is cited separately and immediately contrasted with the weighted-LSE operator.
- Removed citation clustering for the main recent-risk / sparse-reward / non-stationary / regularization comparisons.
- Added an explicit operator-level positioning sentence: nearby methods change a different object (risk measure, reward, representation, update protocol, or data regularization), while this paper changes the Bellman operator itself.
- Added a short sentence saying the most direct empirical baselines are classical Bellman methods plus allocation/safety ablations, because those hold the reward, representation, and update schedule fixed.
- Replaced the discussion section's large citation cluster with a short scope paragraph that refers back to the introduction's detailed distinctions.
- Split the Binary Concrete / Gumbel-Softmax citation cluster into two short cited statements.

Files:
- `neurips_selective_temporal_credit_assignment_positioned.tex`
- `neurips_selective_temporal_credit_assignment_positioned.pdf`
