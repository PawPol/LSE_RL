"""Oracle beta schedule contract tests for Phase VIII M4."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.adaptive_beta.schedules import (
    ALL_METHOD_IDS,
    METHOD_ADAPTIVE_BETA,
    METHOD_ADAPTIVE_BETA_NO_CLIP,
    METHOD_ADAPTIVE_MAGNITUDE_ONLY,
    METHOD_ADAPTIVE_SIGN_ONLY,
    METHOD_CONTRACTION_UCB_BETA,
    METHOD_FIXED_NEGATIVE,
    METHOD_FIXED_POSITIVE,
    METHOD_HAND_ADAPTIVE_BETA,
    METHOD_ORACLE_BETA,
    METHOD_RETURN_UCB_BETA,
    METHOD_VANILLA,
    METHOD_WRONG_SIGN,
    OracleBetaSchedule,
    build_schedule,
)


def _arr(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _update(schedule, episode_index: int, *, episode_info=None) -> None:
    schedule.update_after_episode(
        episode_index,
        rewards=_arr([0.25, 0.50]),
        v_next=_arr([0.05, 0.10]),
        episode_info=episode_info,
    )


class _RegimeReadForbidden(dict):
    """Tripwire for non-oracle schedules: assigning is okay, reading is not."""

    def __contains__(self, key) -> bool:  # type: ignore[override]
        raise AssertionError(f"non-oracle schedule read episode_info key {key!r}")

    def __getitem__(self, key):  # type: ignore[override]
        raise AssertionError(f"non-oracle schedule read episode_info key {key!r}")

    def get(self, key, default=None):  # type: ignore[override]
        raise AssertionError(f"non-oracle schedule read episode_info key {key!r}")

    def keys(self):  # type: ignore[override]
        raise AssertionError("non-oracle schedule inspected episode_info keys")


def test_oracle_in_all_method_ids() -> None:
    assert METHOD_ORACLE_BETA in ALL_METHOD_IDS


def test_oracle_factory_construct() -> None:
    schedule = build_schedule(
        METHOD_ORACLE_BETA,
        env_canonical_sign=None,
        hyperparams={"regime_to_beta": {"plus": 1.0, "minus": -1.0}},
    )

    assert isinstance(schedule, OracleBetaSchedule)
    assert schedule.name == METHOD_ORACLE_BETA


def test_oracle_reads_episode_info_regime() -> None:
    regime_to_beta = {"plus": 1.25, "minus": -0.75}
    schedule = build_schedule(
        METHOD_ORACLE_BETA,
        env_canonical_sign=None,
        hyperparams={"regime_to_beta": regime_to_beta},
    )

    _update(schedule, 0, episode_info={"regime": "plus"})

    assert schedule.beta_for_episode(1) == pytest.approx(regime_to_beta["plus"])


def test_oracle_raises_on_missing_episode_info() -> None:
    schedule = build_schedule(
        METHOD_ORACLE_BETA,
        env_canonical_sign=None,
        hyperparams={"regime_to_beta": {"plus": 1.0}},
    )

    with pytest.raises((KeyError, ValueError)):
        _update(schedule, 0, episode_info=None)


def test_oracle_raises_on_missing_regime_key() -> None:
    schedule = build_schedule(
        METHOD_ORACLE_BETA,
        env_canonical_sign=None,
        hyperparams={"regime_to_beta": {"plus": 1.0}},
    )

    with pytest.raises((KeyError, ValueError)):
        _update(schedule, 0, episode_info={"foo": "bar"})


def test_oracle_raises_on_unknown_regime() -> None:
    schedule = build_schedule(
        METHOD_ORACLE_BETA,
        env_canonical_sign=None,
        hyperparams={"regime_to_beta": {"plus": 1.0}},
    )

    with pytest.raises((KeyError, ValueError)):
        _update(schedule, 0, episode_info={"regime": "minus"})


def test_oracle_is_only_schedule_reading_regime() -> None:
    non_oracle_methods = [
        (METHOD_VANILLA, None, {}),
        (METHOD_FIXED_POSITIVE, None, {}),
        (METHOD_FIXED_NEGATIVE, None, {}),
        (METHOD_WRONG_SIGN, "+", {}),
        (METHOD_ADAPTIVE_BETA, None, {}),
        (METHOD_ADAPTIVE_BETA_NO_CLIP, None, {}),
        (METHOD_ADAPTIVE_SIGN_ONLY, None, {}),
        (METHOD_ADAPTIVE_MAGNITUDE_ONLY, "+", {}),
        (METHOD_HAND_ADAPTIVE_BETA, None, {}),
        (METHOD_CONTRACTION_UCB_BETA, None, {}),
        (METHOD_RETURN_UCB_BETA, None, {}),
    ]
    assert {method for method, _, _ in non_oracle_methods} == (
        set(ALL_METHOD_IDS) - {METHOD_ORACLE_BETA}
    )

    for method_id, env_canonical_sign, hyperparams in non_oracle_methods:
        baseline = build_schedule(method_id, env_canonical_sign, hyperparams)
        with_info = build_schedule(method_id, env_canonical_sign, hyperparams)

        assert with_info.beta_for_episode(0) == pytest.approx(
            baseline.beta_for_episode(0)
        )
        _update(baseline, 0, episode_info=None)
        _update(with_info, 0, episode_info=_RegimeReadForbidden())

        assert with_info.beta_for_episode(1) == pytest.approx(
            baseline.beta_for_episode(1)
        )


def test_oracle_no_smoothing() -> None:
    regime_to_beta = {"plus": 1.5, "minus": -1.5}
    schedule = build_schedule(
        METHOD_ORACLE_BETA,
        env_canonical_sign=None,
        hyperparams={"regime_to_beta": regime_to_beta},
    )

    schedule.update_after_episode(
        0,
        rewards=_arr([100.0, 100.0]),
        v_next=_arr([0.0, 0.0]),
        episode_info={"regime": "plus"},
    )
    first = schedule.beta_for_episode(1)
    schedule.update_after_episode(
        1,
        rewards=_arr([-100.0, -100.0]),
        v_next=_arr([0.0, 0.0]),
        episode_info={"regime": "plus"},
    )
    second = schedule.beta_for_episode(2)

    assert first == pytest.approx(regime_to_beta["plus"])
    assert second == pytest.approx(first)
