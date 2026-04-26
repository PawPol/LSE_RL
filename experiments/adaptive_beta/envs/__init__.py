"""Phase VII environments package.

All Phase VII envs are MushroomRL ``Environment`` subclasses per spec §22.2
resolution. Plain Python helpers (e.g. opponent controllers) live alongside
but are not the primary env API.

Modules
-------
- :mod:`experiments.adaptive_beta.envs.rps` — adversarial Rock-Paper-Scissors
  (M2.1, spec §6.1).
- :mod:`experiments.adaptive_beta.envs.switching_bandit` — 5-arm Bernoulli
  switching bandit (M2.2, spec §6.2 + §22.5; performance-only — excluded
  from mechanism panels at horizon = 1).
- :mod:`experiments.adaptive_beta.envs.hazard_gridworld` — 7x7 gridworld
  with periodically-reshuffling hazards (M2.3, spec §6.3); canonical
  sign ``-`` (pessimistic propagation).
- :mod:`experiments.adaptive_beta.envs.delayed_chain` — chain-of-20
  delayed-reward env (M2.4, spec §6.4); canonical sign ``+``
  (optimistic propagation); deterministic, no shifts.

All envs share the common Phase VII contract (spec §6):

- ``reset(state=None) -> (state, info)`` per MushroomRL convention.
- ``step(action) -> (next_state, reward, absorbing, info)``.
- Numpy state-scalar reads use ``int(np.asarray(x).flat[0])``
  (``tasks/lessons.md`` numpy_state pattern).
- Class attribute ``env_canonical_sign: Optional[str]``; ``None`` for both
  RPS and switching bandit (no canonical sign — ``wrong_sign`` /
  ``adaptive_magnitude_only`` will fail-fast against these envs).
- Property ``current_phase`` (str | int) for diagnostics + the schema
  ``phase`` column.
- Method ``oracle_action() -> Optional[int]`` returning the best-response
  action under the *true* current opponent / arm distribution (regret
  only; the agent never sees this).
- Method ``is_shift_step() -> bool`` returning True at the first step of an
  episode that immediately follows a phase / arm switch (used for the
  schema column ``shift_event``).
- The ``info`` dict at ``step()`` time includes the keys mandated by spec
  §6: ``phase``, ``is_shift_step``, ``oracle_action``, ``catastrophe``
  (False for both envs here), ``terminal_success``.
"""
