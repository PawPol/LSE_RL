"""algorithms subpackage of lse_rl.

Phase IV-C safe weighted-LSE online TD variants:

- :class:`SafeDoubleQLearning`      (two Q-tables, random selection/eval)
- :class:`SafeTargetQLearning`      (online + frozen target, greedy bootstrap)
- :class:`SafeTargetExpectedSARSA`  (online + frozen target, expected bootstrap)

All three classes are standalone (non-MushroomRL-agent) wrappers around
``SafeWeightedCommon``; the Phase IV-C runner is responsible for driving
them through the environment loop.
"""

from .safe_double_q import SafeDoubleQLearning
from .safe_target_expected_sarsa import SafeTargetExpectedSARSA
from .safe_target_q import SafeTargetQLearning

__all__ = [
    "SafeDoubleQLearning",
    "SafeTargetQLearning",
    "SafeTargetExpectedSARSA",
]
