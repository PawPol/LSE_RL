"""Phase IX M_AC_UCB pilot driver — 24-way parallel over (seed × subcase).

Total: 5 seeds × 3 cells = 15 shards; each shard runs 4 γ × 8 methods = 32 runs.
With 15-way concurrent execution, expected wall ≈ 32 × ~40s = ~22 min on a
24-core box (room left for the existing bvo604bzu redispatch on 1 core).

Each shard writes to the same `raw/VIII/M_AC_UCB_pilot/<game>/<cell>/.../` tree;
no path collision because (cell, method, gamma, seed) is unique per shard.

Each shard's stdout/stderr → /tmp/m_ac_ucb_pilot_seed{S}_cell{C}.log.
Master log → /tmp/m_ac_ucb_pilot_master.log.
"""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path('/Users/liq/Documents/Claude/Projects/LSE_RL')
CONFIG = REPO / 'experiments/adaptive_beta/tab_six_games/configs/M_AC_UCB_pilot.yaml'
OUTPUT_ROOT = REPO / 'results/adaptive_beta/tab_six_games'
RUNNER = 'experiments.adaptive_beta.tab_six_games.runners.run_phase_VIII_stage5_adaptive'
LOGDIR = Path('/tmp')
MAX_WORKERS = 15  # 5 seeds × 3 cells

SEEDS = [0, 1, 2, 3, 4]
CELLS = ['DC-Long50', 'AC-Trap', 'SH-FiniteMemoryRegret']


def run_shard(seed: int, cell: str) -> tuple[int, str, int, float]:
    """Spawn one runner subprocess; return (seed, cell, exit_code, wall_seconds)."""
    log_path = LOGDIR / f'm_ac_ucb_pilot_seed{seed}_cell{cell}.log'
    cmd = [
        '.venv/bin/python', '-m', RUNNER,
        '--config', str(CONFIG),
        '--seed-list', str(seed),
        '--subcases', cell,
        '--output-root', str(OUTPUT_ROOT),
    ]
    t0 = time.time()
    with open(log_path, 'w') as logf:
        proc = subprocess.run(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
    return seed, cell, proc.returncode, time.time() - t0


def main() -> int:
    master_log = LOGDIR / 'm_ac_ucb_pilot_master.log'
    with open(master_log, 'w') as ml:
        ml.write(f'M_AC_UCB pilot dispatch start: {dt.datetime.utcnow().isoformat()}Z\n')
        ml.write(f'shards = {len(SEEDS) * len(CELLS)} (5 seeds × 3 cells); max_workers={MAX_WORKERS}\n')
        ml.flush()

        t0 = time.time()
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {
                ex.submit(run_shard, s, c): (s, c)
                for s in SEEDS for c in CELLS
            }
            for fut in concurrent.futures.as_completed(futures):
                s, c, rc, wall = fut.result()
                tag = '✓' if rc == 0 else f'✗ exit={rc}'
                line = f'  shard seed={s} cell={c}: {tag} (wall={wall:.1f}s)'
                ml.write(line + '\n'); ml.flush()
                print(line, flush=True)
                results.append((s, c, rc, wall))

        total_wall = time.time() - t0
        ok = sum(1 for _, _, rc, _ in results if rc == 0)
        bad = len(results) - ok
        ml.write(f'\nM_AC_UCB pilot dispatch finished at {dt.datetime.utcnow().isoformat()}Z\n')
        ml.write(f'OK={ok}/{len(results)}, FAIL={bad}, total_wall={total_wall:.1f}s\n')
        if bad:
            ml.write('\nFailing shards:\n')
            for s, c, rc, _ in results:
                if rc != 0:
                    ml.write(f'  seed={s} cell={c}: exit={rc}; see /tmp/m_ac_ucb_pilot_seed{s}_cell{c}.log\n')
        print(f'\nM_AC_UCB pilot done. OK={ok}/{len(results)}, FAIL={bad}, wall={total_wall:.1f}s')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
