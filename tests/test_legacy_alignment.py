"""Deterministic comparison with the *actual* legacy AF-MADRL simulator.

This deliberately tests common FCFS physics and records where original AoI
accounting differs from the independent PettingZoo benchmark. It is not a
claim of paper-level equivalence.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mec_env_test import Offload
from pettingzoo_mec import MECConfig, env


class FixedProcessingTime:
    def __init__(self, sequence: list[float]) -> None:
        self.sequence = iter(sequence)

    def exponential(self, mean: float) -> float:
        assert mean > 0
        return next(self.sequence)


def action(offload: int = 1, wait: float = 3.0) -> dict:
    return {"offload": offload, "wait": np.asarray([wait], dtype=np.float32)}


def test_legacy_vs_aec_fcfs_timeline_and_aoi_definition() -> None:
    args = SimpleNamespace(
        cuda="cpu", num_iot=2, num_edge=1, num_time=10,
        task_size=8.0, gamma=5.0, LOGNORMAL=0, LOG_VARIANCE=1.0,
        FRACTION=1, STATE_STILL=0, comp_iot=2.5,
        comp_edge=41.8, comp_density=0.3,
    )
    legacy = Offload(args)
    legacy_times = iter([2.0, 1.0])
    legacy.edge_process = lambda size, capacity, density: (next(legacy_times), 1.0)
    legacy.reset(None)
    assert legacy.action_store.tolist() == [1, 1]  # initial auto-offloading
    assert legacy.queue_length_edge.tolist() == [2]
    assert legacy.current_time == 0

    # Real legacy first queued task: t=2; queued service duration released from
    # its surrogate queue_limit multiplier.
    legacy.render()
    assert legacy.current_iot == 0
    assert legacy.current_time == pytest.approx(2.0)
    assert legacy.queue_length_edge.tolist() == [1]
    assert legacy.aoi_average[0] == pytest.approx(0.0)

    # Prevent the completed device from generating another task before t=3.
    legacy.execute_wait(10.0)
    legacy.render()
    assert legacy.current_iot == 1
    assert legacy.current_time == pytest.approx(3.0)
    assert legacy.queue_length_edge.tolist() == [0]
    assert np.allclose(legacy.aoi_average, [0.0, 0.0])

    aec = env(config=MECConfig(n_iot=2, n_edge=1, max_time=10.0))
    aec.reset(seed=17)
    aec._rng = FixedProcessingTime([2.0, 1.0])
    assert aec.agent_selection == "iot_0"
    aec.step(action(offload=1))
    assert aec.time == pytest.approx(0.0)
    assert aec.agent_selection == "iot_1"
    aec.step(action(offload=1))

    # Same completion time and queue occupancy as legacy, despite different
    # initial task setup and event-queue representations.
    assert aec.time == pytest.approx(2.0)
    assert aec.agent_selection == "iot_0"
    assert aec._edge_completions[0] == pytest.approx([3.0])
    assert aec.observe("iot_0")[0] == pytest.approx(1.0)

    aec.step(action(wait=3.0))
    assert aec.time == pytest.approx(3.0)
    assert aec.agent_selection == "iot_1"
    assert aec._edge_completions[0] == []
    assert aec.observe("iot_1")[0] == pytest.approx(0.0)

    # Deliberate difference: AEC integrates a positive AoI area from t=0
    # while the original first completed-update trapezoid terms cancel out.
    assert aec.mean_aoi == pytest.approx(1.5)
    assert aec.mean_aoi != pytest.approx(float(np.mean(legacy.aoi_average)))
    aec.close()
