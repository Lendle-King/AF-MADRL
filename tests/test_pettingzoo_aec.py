"""Official PettingZoo conformance tests plus deterministic physics checks."""
from __future__ import annotations

import numpy as np
import pytest
from pettingzoo.test import api_test, seed_test

from pettingzoo_mec import MECConfig, env


def test_official_pettingzoo_api() -> None:
    simulation = env(config=MECConfig(n_iot=3, n_edge=2, max_time=20.0))
    api_test(simulation, num_cycles=250, verbose_progress=False)
    simulation.close()


def test_official_pettingzoo_seeding() -> None:
    seed_test(lambda: env(config=MECConfig(n_iot=3, n_edge=2, max_time=20.0)), num_cycles=20)


def sample_trace(seed: int) -> list[tuple[str, float, float, float]]:
    simulation = env(config=MECConfig(n_iot=3, n_edge=2, max_time=8.0))
    simulation.reset(seed=seed)
    trace: list[tuple[str, float, float, float]] = []
    for agent in simulation.agent_iter():
        obs, reward, terminated, truncated, info = simulation.last()
        trace.append((agent, round(simulation.time, 8), round(float(reward), 8), simulation.mean_aoi))
        simulation.step(None if terminated or truncated else simulation.action_space(agent).sample())
    assert simulation.agents == []
    simulation.close()
    return trace


def test_seeded_event_trace_is_deterministic() -> None:
    assert sample_trace(81) == sample_trace(81)


def test_horizon_rewards_and_spaces() -> None:
    simulation = env(config=MECConfig(n_iot=2, n_edge=1, max_time=4.0))
    simulation.reset(seed=17)
    last_time = simulation.time
    actions = 0
    for agent in simulation.agent_iter():
        observation, reward, terminated, truncated, info = simulation.last()
        assert simulation.observation_space(agent).contains(observation)
        assert np.isfinite(reward)
        assert simulation.time >= last_time
        assert simulation.time <= simulation.config.max_time
        last_time = simulation.time
        if terminated or truncated:
            assert truncated  # Time horizon is a truncation, not task termination.
            simulation.step(None)
        else:
            action = simulation.action_space(agent).sample()
            action["wait"] = np.asarray([0.5], dtype=np.float32)
            simulation.step(action)
            actions += 1

    assert actions > 0
    assert simulation.agents == []
    assert simulation.time == pytest.approx(4.0)
    assert simulation.mean_aoi > 0
    assert simulation._aoi_areas.sum() > 0
    simulation.close()


def test_aoi_area_single_agent_before_first_completion() -> None:
    simulation = env(config=MECConfig(n_iot=1, n_edge=1, max_time=0.000001))
    simulation.reset(seed=1)
    action = {"offload": 0, "wait": np.asarray([0.5], dtype=np.float32)}
    simulation.step(action)
    # By construction a generated update starts at t=0. Until the first
    # completion, age(t)=t and its exact integral is max_time**2 / 2.
    assert simulation.time == pytest.approx(simulation.config.max_time)
    assert simulation._aoi_areas[0] == pytest.approx(0.5 * simulation.time ** 2)
    while simulation.agents:
        simulation.step(None)
    simulation.close()


def test_reject_invalid_action() -> None:
    simulation = env(config=MECConfig(n_iot=2, n_edge=1))
    simulation.reset(seed=5)
    with pytest.raises(ValueError):
        simulation.step({"offload": 999, "wait": np.asarray([1], dtype=np.float32)})
    simulation.close()


def test_edge_fcfs_queue() -> None:
    simulation = env(config=MECConfig(n_iot=2, n_edge=1, max_time=50))
    simulation.reset(seed=123)
    action = {"offload": 1, "wait": np.asarray([0.0], dtype=np.float32)}
    assert simulation.agent_selection == "iot_0"
    simulation.step(action)
    first_available = simulation._edge_available_until[0]
    assert simulation.agent_selection == "iot_1"
    simulation.step(action)
    assert simulation._edge_available_until[0] > first_available
    assert simulation._last_latency[1] >= first_available
    simulation.close()
