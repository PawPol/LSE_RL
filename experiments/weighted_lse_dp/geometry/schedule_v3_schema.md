# Schedule v3 JSON Schema

Phase IV-A stagewise natural-shift-first calibration schedule.

## Required Fields

| Field                  | Type          | Shape / Value         | Description                                                              |
|------------------------|---------------|-----------------------|--------------------------------------------------------------------------|
| `phase`                | string        | `"phase4"`            | Phase identifier.                                                        |
| `schedule_version`     | int           | `3`                   | Schema version.                                                          |
| `task_family`          | string        | ---                   | Task family name (e.g. `"chain_jackpot"`).                               |
| `scheduler_mode`       | string        | `"stagewise_u"`       | Always `"stagewise_u"` for v3.                                           |
| `gamma_eval`           | float         | scalar                | Evaluation discount factor.                                              |
| `gamma_base`           | float         | scalar                | Operator baseline discount factor.                                       |
| `sign_family`          | int           | +1 or -1              | Operator family sign.                                                    |
| `reward_bound`         | float         | scalar                | One-step reward bound r_max.                                             |
| `alpha_t`              | list[float]   | T                     | Adaptive headroom per stage.                                             |
| `kappa_t`              | list[float]   | T                     | Contraction rate kappa_t = gamma_base + alpha_t * (1 - gamma_base).      |
| `Bhat_t`               | list[float]   | T+1                   | Certified radius. `Bhat_t[T] = 0` (terminal).                           |
| `A_t`                  | list[float]   | T                     | Normalization anchor A_t = r_max + Bhat_{t+1}.                           |
| `xi_ref_t`             | list[float]   | T                     | Reference normalized margin Q_0.75(a_t | a_t > 0), clipped.             |
| `u_target_t`           | list[float]   | T                     | Target natural shift: u_min + (u_max - u_min) * I_t.                    |
| `u_tr_cap_t`           | list[float]   | T                     | Trust-region cap on natural shift.                                       |
| `U_safe_ref_t`         | list[float]   | T                     | Safe-headroom cap from fixed-point iteration.                            |
| `u_ref_used_t`         | list[float]   | T                     | Effective natural shift: min(u_target, u_tr_cap, |U_safe_ref|).         |
| `theta_used_t`         | list[float]   | T                     | Effective theta: sign * u_ref_used / max(xi_ref, xi_floor).             |
| `beta_used_t`          | list[float]   | T                     | Effective beta: theta_used / max(A_t, 1e-8).                            |
| `trust_clip_active_t`  | list[bool]    | T                     | True where trust-region cap was the binding constraint.                  |
| `safe_clip_active_t`   | list[bool]    | T                     | True where safe-headroom cap was binding (and trust was not).            |
| `source_phase`         | string        | ---                   | Source of input data: `"phase3"`, `"phase12"`, or `"pilot"`.             |
| `notes`                | string        | ---                   | Free-text notes.                                                         |
| `provenance`           | object        | ---                   | Provenance metadata (see below).                                         |

## Provenance Object

| Field                    | Type   | Description                                     |
|--------------------------|--------|-------------------------------------------------|
| `git_sha`                | string | Git commit SHA at schedule build time.           |
| `calibration_code_version` | string | Version string (e.g. `"v3.0"`).               |
| `input_hashes`           | object | SHA-256 prefix hashes of input arrays.           |
| `input_hashes.margins`   | string | Hash of concatenated margins arrays.             |
| `input_hashes.p_align`   | string | Hash of p_align_by_stage array.                  |
| `input_hashes.n_by_stage`| string | Hash of n_by_stage array.                        |

## Array Shapes

- All stagewise arrays (`alpha_t`, `kappa_t`, `A_t`, `xi_ref_t`, `u_target_t`, `u_tr_cap_t`, `U_safe_ref_t`, `u_ref_used_t`, `theta_used_t`, `beta_used_t`, `trust_clip_active_t`, `safe_clip_active_t`) have length **T** (number of stages).
- `Bhat_t` has length **T+1**, with `Bhat_t[T] = 0` as the terminal boundary condition.

## Reading the Schedule from JSON

```python
import json
import numpy as np

with open("schedule.json") as f:
    sched = json.load(f)

beta = np.array(sched["beta_used_t"])  # shape (T,)
bhat = np.array(sched["Bhat_t"])       # shape (T+1,)
T = len(beta)
assert len(bhat) == T + 1
assert sched["schedule_version"] == 3
```

## Example (T=3)

```json
{
  "phase": "phase4",
  "schedule_version": 3,
  "task_family": "chain_jackpot",
  "scheduler_mode": "stagewise_u",
  "gamma_eval": 0.97,
  "gamma_base": 0.95,
  "sign_family": 1,
  "reward_bound": 1.0,
  "alpha_t": [0.10, 0.15, 0.08],
  "kappa_t": [0.955, 0.9575, 0.954],
  "Bhat_t": [12.5, 10.2, 5.1, 0.0],
  "A_t": [11.2, 6.1, 1.0],
  "xi_ref_t": [0.15, 0.22, 0.05],
  "u_target_t": [0.010, 0.015, 0.005],
  "u_tr_cap_t": [0.008, 0.014, 0.004],
  "U_safe_ref_t": [0.012, 0.018, 0.003],
  "u_ref_used_t": [0.008, 0.014, 0.003],
  "theta_used_t": [0.053, 0.064, 0.060],
  "beta_used_t": [0.0047, 0.0105, 0.060],
  "trust_clip_active_t": [true, true, false],
  "safe_clip_active_t": [false, false, true],
  "source_phase": "pilot",
  "notes": "Example schedule for illustration.",
  "provenance": {
    "git_sha": "abc123def456",
    "calibration_code_version": "v3.0",
    "input_hashes": {
      "margins": "a1b2c3d4e5f6g7h8",
      "p_align": "1234567890abcdef",
      "n_by_stage": "fedcba0987654321"
    }
  }
}
```
