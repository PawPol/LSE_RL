# Phase VIII M5 W2 handoff - schema parity + figure smoke tests

## Summary

- Added schema-parity regression tests for the Phase VIII long-CSV
  aggregator contract from spec §8.3 and lessons.md #11/#16/#19/#20.
- Added parametrized figure smoke tests for all five M5 W1.B plotting
  modules, using an 8-row synthetic long CSV with all 49 columns.
- Verified the required test target with the repo virtual environment.

## Artifacts

- `tests/adaptive_beta/tab_six_games/test_aggregate_schema_parity.py`
- `tests/adaptive_beta/tab_six_games/test_figure_smoke.py`
- `results/adaptive_beta/tab_six_games/M5_W2_handoff.md`

No edits were made to `src/`, `mushroom-rl-dev/`, or `experiments/`.
No package installs were performed.

## Verification evidence

Command:

```text
.venv/bin/python -m pytest tests/adaptive_beta/tab_six_games/test_aggregate_schema_parity.py tests/adaptive_beta/tab_six_games/test_figure_smoke.py -q
```

Timed command:

```text
/usr/bin/time -p .venv/bin/python -m pytest tests/adaptive_beta/tab_six_games/test_aggregate_schema_parity.py tests/adaptive_beta/tab_six_games/test_figure_smoke.py -q
```

Result:

```text
..............................                                           [100%]
real 11.08
user 10.66
sys 0.31
```

Counts:

- Schema parity tests: 10
- Figure smoke tests: 20
- Total passed: 30
- Total failed: 0
- Runtime: 11.08 seconds

## Open questions

- None.
