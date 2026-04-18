"""
Finite-horizon dynamic programming planners and utilities.

Phase I subpackage. See ``finite_horizon_dp_utils`` for the pure-numpy helpers
consumed by the DP planners (``policy_evaluation``, ``value_iteration``,
``policy_iteration``, ``modified_policy_iteration``, ``async_value_iteration``).
"""
from mushroom_rl.algorithms.value.dp.finite_horizon_dp_utils import (
    validate_finite_mdp,
    extract_mdp_arrays,
    expected_reward,
    allocate_value_tables,
    bellman_q_backup,
    bellman_v_from_q,
    greedy_policy,
    bellman_q_policy_backup,
    sup_norm_residual,
    deterministic_policy_array,
)
from mushroom_rl.algorithms.value.dp.classical_policy_evaluation import (
    ClassicalPolicyEvaluation,
)
from mushroom_rl.algorithms.value.dp.classical_value_iteration import (
    ClassicalValueIteration,
)
from mushroom_rl.algorithms.value.dp.classical_policy_iteration import (
    ClassicalPolicyIteration,
)
from mushroom_rl.algorithms.value.dp.classical_modified_policy_iteration import (
    ClassicalModifiedPolicyIteration,
)
from mushroom_rl.algorithms.value.dp.classical_async_value_iteration import (
    ClassicalAsyncValueIteration,
)
from mushroom_rl.algorithms.value.dp.safe_weighted_common import (
    BetaSchedule,
    SafeWeightedCommon,
)
from mushroom_rl.algorithms.value.dp.safe_weighted_value_iteration import (
    SafeWeightedValueIteration,
)
from mushroom_rl.algorithms.value.dp.safe_weighted_policy_evaluation import (
    SafeWeightedPolicyEvaluation,
)
from mushroom_rl.algorithms.value.dp.safe_weighted_policy_iteration import (
    SafeWeightedPolicyIteration,
)
from mushroom_rl.algorithms.value.dp.safe_weighted_modified_policy_iteration import (
    SafeWeightedModifiedPolicyIteration,
)
from mushroom_rl.algorithms.value.dp.safe_weighted_async_value_iteration import (
    SafeWeightedAsyncValueIteration,
)

__all__ = [
    "validate_finite_mdp",
    "extract_mdp_arrays",
    "expected_reward",
    "allocate_value_tables",
    "bellman_q_backup",
    "bellman_v_from_q",
    "greedy_policy",
    "bellman_q_policy_backup",
    "sup_norm_residual",
    "deterministic_policy_array",
    "ClassicalPolicyEvaluation",
    "ClassicalValueIteration",
    "ClassicalPolicyIteration",
    "ClassicalModifiedPolicyIteration",
    "ClassicalAsyncValueIteration",
    "BetaSchedule",
    "SafeWeightedCommon",
    "SafeWeightedValueIteration",
    "SafeWeightedPolicyEvaluation",
    "SafeWeightedPolicyIteration",
    "SafeWeightedModifiedPolicyIteration",
    "SafeWeightedAsyncValueIteration",
]
