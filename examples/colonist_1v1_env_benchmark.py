#!/usr/bin/env python3
"""Measure Colonist environment throughput for DummyVecEnv and SubprocVecEnv."""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

try:
    from examples.colonist_1v1_train import make_colonist_env
except ModuleNotFoundError:  # Direct execution: sys.path starts at examples/.
    from colonist_1v1_train import make_colonist_env


def _build_env(seed: int):
    return make_colonist_env(seed=seed, randomize_seats=True)


def benchmark_mode(
    mode: str,
    *,
    n_envs: int,
    steps: int,
    seed: int,
    start_method: str | None,
) -> dict:
    factories = [partial(_build_env, seed + rank) for rank in range(n_envs)]
    if mode == "subproc":
        env = SubprocVecEnv(factories, start_method=start_method)
    else:
        env = DummyVecEnv(factories)
    try:
        env.reset()
        started = time.perf_counter()
        for _ in range(steps):
            masks = get_action_masks(env)
            actions = np.asarray(
                [int(np.flatnonzero(mask)[0]) for mask in masks], dtype=np.int64
            )
            env.step(actions)
        elapsed = time.perf_counter() - started
    finally:
        env.close()
    decisions = steps * n_envs
    return {
        "mode": mode,
        "n_envs": n_envs,
        "steps": steps,
        "decisions": decisions,
        "elapsed_seconds": elapsed,
        "decisions_per_second": decisions / elapsed,
        "seed": seed,
        "start_method": start_method if mode == "subproc" else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--modes", nargs="+", choices=("dummy", "subproc"), default=("dummy", "subproc")
    )
    parser.add_argument("--start-method", default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.n_envs <= 0 or args.steps <= 0:
        parser.error("--n-envs and --steps must be positive")

    results = [
        benchmark_mode(
            mode,
            n_envs=args.n_envs,
            steps=args.steps,
            seed=args.seed,
            start_method=args.start_method,
        )
        for mode in args.modes
    ]
    print(f"{'mode':10s} {'envs':>5s} {'decisions/s':>14s} {'elapsed':>10s}")
    for row in results:
        print(
            f"{row['mode']:10s} {row['n_envs']:5d} "
            f"{row['decisions_per_second']:14,.1f} "
            f"{row['elapsed_seconds']:9.3f}s"
        )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"results": results}, indent=2))
        print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
