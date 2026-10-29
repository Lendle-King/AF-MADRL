"""Shared Dueling Double DQN offloading baseline for the AEC AoI-MEC benchmark.

Waiting is intentionally fixed; only offloading decisions are learned. Rewards
are aggregated across the two AEC phases until each agent's next OFFLOAD
decision, yielding valid SMDP transitions with actual elapsed time.

Usage:
    python examples/train_aec.py --episodes 5 --horizon 20 --seed 42
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from pettingzoo_mec import MECConfig, env
from pettingzoo_mec.agents import DoubleDQNAgent, ReplayBuffer, TimedTransition


def run_training(
    *,
    episodes: int = 5,
    horizon: float = 20.0,
    n_iot: int = 4,
    n_edge: int = 2,
    fixed_wait: float = 1.0,
    batch_size: int = 32,
    seed: int = 42,
) -> list[float]:
    if episodes < 1 or batch_size < 1:
        raise ValueError("episodes and batch_size must be positive")
    config = MECConfig(n_iot=n_iot, n_edge=n_edge, max_time=horizon)
    if not 0.0 <= fixed_wait <= config.wait_max:
        raise ValueError("fixed_wait outside action space")
    torch.manual_seed(seed)
    simulation = env(config=config)
    learner = DoubleDQNAgent(
        observation_dim=n_edge + 5, action_dim=n_edge + 1,
        gamma=0.99, reference_time=1.0, seed=seed,
    )
    replay = ReplayBuffer(seed=seed)
    results: list[float] = []

    for episode in range(episodes):
        simulation.reset(seed=seed + episode)
        # Each IoT can have at most one pending offload transition.
        pending: dict[str, tuple[np.ndarray, int, float]] = {}
        accrued: dict[str, float] = {a: 0.0 for a in simulation.possible_agents}
        epsilon = max(0.05, 0.8 * (0.85 ** episode))
        losses: list[float] = []

        for agent in simulation.agent_iter():
            observation, reward, terminated, truncated, info = simulation.last()
            accrued[agent] += float(reward)
            if terminated or truncated:
                if agent in pending:
                    old_obs, old_action, old_time = pending.pop(agent)
                    replay.add(
                        TimedTransition(
                            old_obs, old_action, accrued[agent],
                            observation.copy(), bool(terminated), simulation.time - old_time,
                        )
                    )
                simulation.step(None)
                continue

            action = simulation.action_space(agent).sample()
            if info["phase"] == "offload":
                if agent in pending:
                    old_obs, old_action, old_time = pending.pop(agent)
                    replay.add(
                        TimedTransition(
                            old_obs, old_action, accrued[agent],
                            observation.copy(), False, simulation.time - old_time,
                        )
                    )
                accrued[agent] = 0.0
                offload_choice = learner.act(observation, epsilon)
                action["offload"] = offload_choice
                pending[agent] = (observation.copy(), offload_choice, simulation.time)
            else:
                action["wait"] = np.asarray([fixed_wait], dtype=np.float32)

            simulation.step(action)
            if len(replay) >= batch_size:
                losses.append(learner.learn(replay.sample(batch_size)))

        assert not pending, "offload transitions were not flushed at the episode boundary"
        results.append(simulation.mean_aoi)
        loss_text = f"{np.mean(losses):.5f}" if losses else "n/a"
        print(
            f"episode={episode + 1} mean_aoi={simulation.mean_aoi:.4f} "
            f"replay={len(replay)} epsilon={epsilon:.3f} loss={loss_text}"
        )
    simulation.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--horizon", type=float, default=20.0)
    parser.add_argument("--n-iot", type=int, default=4)
    parser.add_argument("--n-edge", type=int, default=2)
    parser.add_argument("--fixed-wait", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_training(
        episodes=args.episodes, horizon=args.horizon,
        n_iot=args.n_iot, n_edge=args.n_edge,
        fixed_wait=args.fixed_wait, batch_size=args.batch_size, seed=args.seed,
    )


if __name__ == "__main__":
    main()
