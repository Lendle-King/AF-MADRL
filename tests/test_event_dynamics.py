"""Independent event-physics, causal-observation and reward-ledger checks.

These tests deliberately go beyond PettingZoo's public API checks: API tests
alone do not verify whether the simulator's scientific quantities are correct.
"""
from __future__ import annotations

import numpy as np
import pytest

from pettingzoo_mec import MECConfig, env


class FixedServiceTimes:
    """Inject deterministic durations into the simulator (test-only)."""

    def __init__(self, durations: list[float]):
        self.durations = iter(durations)

    def exponential(self, mean: float) -> float:
        assert mean > 0
        return next(self.durations)


def make_action(offload: int = 0, wait: float = 1.0) -> dict:
    return {"offload": offload, "wait": np.array([wait], dtype=np.float32)}


def test_exact_aoi_integral_and_aec_reward_ledger() -> None:
    """Hand-computed trajectory: ages, rewards, latency and horizon agree."""
    sim = env(config=MECConfig(n_iot=1, n_edge=1, max_time=6.0, reward_scale=100.0))
    sim.reset(seed=1)
    sim._rng = FixedServiceTimes([2.0, 1.0])

    observed_rewards: list[float] = []
    assert sim.agent_selection == "iot_0"
    sim.step(make_action(offload=0))
    observed_rewards.append(sim.last()[1])
    assert sim.time == pytest.approx(2.0)
    assert sim._aoi_areas[0] == pytest.approx(2.0)
    assert sim.observe("iot_0")[-2] == pytest.approx(2.0)
    assert sim.infos["iot_0"]["phase"] == "wait"

    sim.step(make_action(wait=1.0))
    observed_rewards.append(sim.last()[1])
    assert sim.time == pytest.approx(3.0)
    assert sim._aoi_areas[0] == pytest.approx(4.5)
    assert sim.infos["iot_0"]["phase"] == "offload"

    sim.step(make_action(offload=0))
    observed_rewards.append(sim.last()[1])
    assert sim.time == pytest.approx(4.0)
    assert sim._aoi_areas[0] == pytest.approx(8.0)
    assert sim.observe("iot_0")[-2] == pytest.approx(1.0)
    assert sim.infos["iot_0"]["instantaneous_aoi"] == pytest.approx(1.0)

    sim.step(make_action(wait=2.0))
    assert sim.time == pytest.approx(6.0)
    assert sim.truncations["iot_0"]
    observed_rewards.append(sim.last()[1])
    assert sim._aoi_areas[0] == pytest.approx(12.0)
    assert sim.mean_aoi == pytest.approx(2.0)
    assert observed_rewards == pytest.approx([-0.02, -0.025, -0.035, -0.04])
    assert sum(observed_rewards) == pytest.approx(-sim._aoi_areas.sum() / 100.0)
    sim.step(None)
    assert sim.agents == []
    sim.close()


def test_inflight_service_time_is_not_revealed_to_other_agents() -> None:
    """CTDE state and individual observations must not reveal future latency."""
    sim = env(config=MECConfig(n_iot=2, n_edge=1, max_time=10.0))
    sim.reset(seed=8)
    sim._rng = FixedServiceTimes([2.0, 0.5])
    assert sim.agent_selection == "iot_0"
    sim.step(make_action(offload=1))
    assert sim.time == 0
    assert sim.agent_selection == "iot_1"
    assert sim.observe("iot_0")[-2] == 0.0
    assert sim.state()[4] == 0.0  # first agent's last-observed latency
    assert np.all(sim._last_latency == 0)

    sim.step(make_action(offload=0))
    assert sim.time == pytest.approx(0.5)
    assert sim.agent_selection == "iot_1"
    assert sim.infos["iot_1"]["phase"] == "wait"
    assert sim.observe("iot_1")[-2] == pytest.approx(0.5)
    assert sim.observe("iot_0")[-2] == 0.0  # still in flight
    sim.step(make_action(wait=2.0))
    assert sim.time == pytest.approx(2.0)
    assert sim.agent_selection == "iot_0"
    assert sim.observe("iot_0")[-2] == pytest.approx(2.0)
    sim.close()


def test_deterministic_edge_queue_respects_fcfs_completion_time() -> None:
    sim = env(config=MECConfig(n_iot=2, n_edge=1, max_time=10.0))
    sim.reset(seed=8)
    sim._rng = FixedServiceTimes([2.0, 1.0])

    sim.step(make_action(offload=1))  # iot_0: service 0 -> 2
    assert sim.agent_selection == "iot_1"
    sim.step(make_action(offload=1))  # iot_1: queue at 0; service 2 -> 3
    assert sim.agent_selection == "iot_0"
    assert sim.time == pytest.approx(2.0)
    assert sim._edge_available_until[0] == pytest.approx(3.0)
    assert sim._last_latency[0] == pytest.approx(2.0)
    assert sim._last_latency[1] == 0.0
    assert sim.observe("iot_1")[0] == pytest.approx(1.0)

    sim.step(make_action(wait=1.0))
    assert sim.time == pytest.approx(3.0)
    assert sim._last_latency[1] == pytest.approx(3.0)
    assert sim._edge_completions[0] == []
    sim.close()


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_reject_invalid_nonfinite_physical_parameters(bad: float) -> None:
    with pytest.raises(ValueError):
        MECConfig(max_time=bad)


@pytest.mark.parametrize("n_iot,n_edge", [(1, 1), (2, 1), (3, 2), (5, 3)])
@pytest.mark.parametrize("seed", [0, 7, 24])
@pytest.mark.parametrize("horizon", [1.0, 4.0])
def test_randomized_event_reward_conservation(
    n_iot: int, n_edge: int, seed: int, horizon: float
) -> None:
    """Across varied topologies, all rewards equal exactly the global AoI area."""
    cfg = MECConfig(n_iot=n_iot, n_edge=n_edge, max_time=horizon)
    sim = env(config=cfg)
    sim.reset(seed=seed)
    total_received = 0.0
    last_time = 0.0
    last_areas = np.zeros(n_iot, dtype=np.float64)
    last_generations = np.zeros(n_iot, dtype=np.float64)
    steps = 0

    for agent in sim.agent_iter(50000):
        observation, reward, terminated, truncated, info = sim.last()
        assert sim.observation_space(agent).contains(observation)
        assert np.isfinite(reward) and np.isfinite(sim.mean_aoi)
        assert last_time <= sim.time <= horizon
        assert np.all(sim._aoi_areas >= last_areas)
        assert np.all(sim._last_delivered_generation >= last_generations)
        assert np.all(sim._last_delivered_generation <= sim.time)
        total_received += float(reward)
        last_time = sim.time
        last_areas = sim._aoi_areas.copy()
        last_generations = sim._last_delivered_generation.copy()

        if terminated or truncated:
            sim.step(None)
        else:
            action = sim.action_space(agent).sample()
            if info["phase"] == "wait":
                # Include zero-wait and positive-wait events.
                action["wait"] = np.array([0.0 if steps % 7 == 0 else 0.5], dtype=np.float32)
            sim.step(action)
        steps += 1

    assert sim.agents == [], "failed to reach horizon within 50,000 steps"
    assert steps > n_iot
    assert sim.time == pytest.approx(horizon)
    assert total_received == pytest.approx(
        float(-sim._aoi_areas.sum() / cfg.reward_scale), abs=1e-7, rel=1e-9
    )
    sim.close()
