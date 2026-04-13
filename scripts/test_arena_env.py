"""Smoke test: run SO-100 Arena environment with zero-action policy.

Usage:
    <ISAAC_SIM>/python.sh scripts/test_arena_env.py --headless --num_envs 1

Or via Arena's policy_runner.py:
    <ISAAC_SIM>/python.sh <ARENA>/isaaclab_arena/examples/policy_runner.py \
        --environment isaaclab_arena_vla.environments.so100_pick_and_place:SO100PickAndPlaceEnvironment \
        so100_pick_and_place --enable_cameras
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="SO-100 Arena env smoke test")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=100)
    args = parser.parse_args()

    # ─── Phase 1: Start SimulationApp ─────────────────────────────────────
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})

    # ─── Phase 2: Import Arena components (requires SimulationApp) ────────
    import torch
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy

    from isaaclab_arena_vla.environments.so100_pick_and_place import SO100PickAndPlaceEnvironment

    # ─── Phase 3: Build environment ──────────────────────────────────────
    env_cls = SO100PickAndPlaceEnvironment()

    # Create a namespace to simulate CLI args
    cli_args = argparse.Namespace(
        enable_cameras=False,
        episode_length=40.0,
        headless=args.headless,
        num_envs=args.num_envs,
        seed=42,
        device="cuda:0",
        disable_fabric=False,
        disable_pinocchio=True,
        mimic=False,
    )

    arena_env = env_cls.get_env(cli_args)
    builder = ArenaEnvBuilder(arena_env, cli_args)
    env = builder.make_registered()

    sys.stderr.write("[test] Environment created successfully.\n")

    # ─── Phase 4: Run with zero-action policy ────────────────────────────
    policy = ZeroActionPolicy()
    obs, _ = env.reset()
    sys.stderr.write("[test] Environment reset OK.\n")

    for step in range(args.num_steps):
        with torch.inference_mode():
            actions = policy.get_action(env, obs)
            obs, _, terminated, truncated, _ = env.step(actions)

            if terminated.any() or truncated.any():
                env_ids = (terminated | truncated).nonzero().flatten()
                policy.reset(env_ids=env_ids)
                sys.stderr.write(f"[test] Episode ended at step {step}\n")

        if step % 20 == 0:
            sys.stderr.write(f"[test] Step {step}/{args.num_steps}\n")

        if not simulation_app.is_running():
            break

    sys.stderr.write(f"[test] Completed {args.num_steps} steps.\n")

    # ─── Phase 5: Metrics ────────────────────────────────────────────────
    try:
        from isaaclab_arena.metrics.metrics import compute_metrics
        metrics = compute_metrics(env)
        sys.stderr.write(f"[test] Metrics: {metrics}\n")
    except Exception as e:
        sys.stderr.write(f"[test] Metrics computation skipped: {e}\n")

    env.close()
    simulation_app.close()
    sys.stderr.write("[test] Done.\n")


if __name__ == "__main__":
    main()
