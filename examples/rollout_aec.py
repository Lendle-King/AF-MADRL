"""Run a random PettingZoo rollout without installing PyTorch.

Usage:
    python examples/rollout_aec.py
"""
from __future__ import annotations

import numpy as np

from pettingzoo_mec import MECConfig, env


def main() -> None:
    simulation = env(config=MECConfig(n_iot=4, n_edge=2, max_time=15.0))
    simulation.reset(seed=42)
    decisions = 0
    for agent in simulation.agent_iter():
        observation, reward, terminated, truncated, info = simulation.last()
        if terminated or truncated:
            simulation.step(None)
            continue
        assert simulation.observation_space(agent).contains(observation)
        action = simulation.action_space(agent).sample()
        if info["phase"] == "wait":
            action["wait"] = np.array([1.0], dtype=np.float32)
        simulation.step(action)
        decisions += 1
    print(f"decisions={decisions}, elapsed={simulation.time:.2f}, mean_aoi={simulation.mean_aoi:.4f}")
    simulation.close()


if __name__ == "__main__":
    main()
