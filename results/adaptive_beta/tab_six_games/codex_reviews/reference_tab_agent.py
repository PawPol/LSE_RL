"""Independent AC-FictitiousPlay TAB Q-learning reference.

No production-code imports. Reimplements the repeated asymmetric
coordination game, finite-memory fictitious-play opponent, epsilon-greedy
TAB Q-learning update, and production-file AUC comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path("results/adaptive_beta/tab_six_games/raw/VIII/stage1_beta_sweep")
BASE = ROOT / "asymmetric_coordination" / "AC-FictitiousPlay"
PAYOFF_AGENT = np.array([[5.0, 0.0], [3.0, 3.0]], dtype=np.float64)
PAYOFF_OPPONENT = PAYOFF_AGENT.T.copy()


def tab_target(beta: float, gamma: float, reward: float, v_next: float) -> float:
    if abs(beta) <= 1.0e-8:
        return float(reward + gamma * v_next)
    log_sum = np.logaddexp(beta * reward, beta * v_next + np.log(gamma))
    return float(((1.0 + gamma) / beta) * (log_sum - np.log(1.0 + gamma)))


def epsilon(episode: int) -> float:
    if episode <= 0:
        return 1.0
    if episode >= 5000:
        return 0.05
    return float(1.0 + (0.05 - 1.0) * (episode / 5000))


def state_id(step_in_episode: int, last_opp_action: int | None) -> int:
    step = min(max(int(step_in_episode), 0), 19)
    slot = 0 if last_opp_action is None else int(last_opp_action) + 1
    return step * 3 + slot


def opponent_action(history: list[int], rng: np.random.Generator) -> int:
    window = history[-20:]
    if not window:
        belief = np.array([0.5, 0.5], dtype=np.float64)
    else:
        counts = np.bincount(np.asarray(window, dtype=np.int64), minlength=2)
        belief = counts.astype(np.float64) / float(counts.sum())
    values = belief @ PAYOFF_OPPONENT
    ties = np.flatnonzero(values >= float(values.max()) - 1.0e-12)
    return int(ties[0]) if ties.size == 1 else int(rng.choice(ties))


def run_reference(beta: float, seed: int, episodes: int = 1000) -> np.ndarray:
    agent_rng = np.random.default_rng(seed)
    adv_rng = np.random.default_rng(seed)
    q = np.zeros((60, 2), dtype=np.float64)
    agent_history: list[int] = []
    returns = np.zeros(episodes, dtype=np.float64)
    for ep in range(episodes):
        s = state_id(0, None)
        ret = 0.0
        for t in range(20):
            if agent_rng.random() < epsilon(ep):
                action = int(agent_rng.integers(0, 2))
            else:
                action = int(np.argmax(q[s]))
            opp = opponent_action(agent_history, adv_rng)
            reward = float(PAYOFF_AGENT[action, opp])
            absorbing = t == 19
            ns = state_id(t + 1, opp)
            v_next = 0.0 if absorbing else float(np.max(q[ns]))
            target = tab_target(beta, 0.95, reward, v_next)
            q[s, action] += 0.1 * (target - q[s, action])
            agent_history.append(action)
            ret += reward
            s = ns
        returns[ep] = ret
    return returns


def production_returns(beta: float, seed: int, episodes: int = 1000) -> np.ndarray:
    method = "vanilla" if beta == 0.0 else f"fixed_beta_{beta:+g}"
    path = BASE / method / f"seed_{seed}" / "metrics.npz"
    with np.load(path, allow_pickle=False) as data:
        returns = np.asarray(data["return"], dtype=np.float64)
    return returns[:episodes]


def main() -> int:
    print("game,beta,seed,reference_auc,production_auc,pct_diff")
    for beta in (-1.0, 0.0, 1.0):
        for seed in (0, 1, 2):
            ref = run_reference(beta, seed)
            prod = production_returns(beta, seed)
            ref_auc = float(np.trapezoid(ref))
            prod_auc = float(np.trapezoid(prod))
            pct = abs(ref_auc - prod_auc) / max(abs(prod_auc), 1.0) * 100.0
            print(
                "AC-FictitiousPlay,"
                f"{beta:+g},{seed},{ref_auc:.6f},{prod_auc:.6f},{pct:.6f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
