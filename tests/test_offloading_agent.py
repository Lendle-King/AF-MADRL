"""Regression checks for action support, terminal masks and SMDP discount."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pettingzoo_mec.agents import (  # noqa: E402
    DoubleDQNAgent,
    ReplayBuffer,
    TimedTransition,
    elapsed_discount,
)


def test_elapsed_discount() -> None:
    assert elapsed_discount(0.9, 0.0) == pytest.approx(1.0)
    assert elapsed_discount(0.9, 2.0) == pytest.approx(0.81)
    assert elapsed_discount(0.9, 2.0, reference_time=2.0) == pytest.approx(0.9)
    with pytest.raises(ValueError):
        elapsed_discount(0.9, -1.0)


def test_epsilon_samples_all_legal_offload_actions() -> None:
    agent = DoubleDQNAgent(7, 4, seed=9)
    observed = {agent.act(np.zeros(7, dtype=np.float32), epsilon=1.0) for _ in range(150)}
    assert observed == {0, 1, 2, 3}


def test_timed_replay_updates_and_syncs_target() -> None:
    learner = DoubleDQNAgent(7, 3, seed=4, target_sync=1)
    buffer = ReplayBuffer(seed=4)
    for i in range(32):
        buffer.add(
            TimedTransition(
                state=np.full(7, i / 32, dtype=np.float32),
                action=i % 3,
                reward=-(i + 1) * 0.1,
                next_state=np.full(7, (i + 1) / 32, dtype=np.float32),
                done=(i == 31),
                elapsed=0.1 + i / 100,
            )
        )
    loss = learner.learn(buffer.sample(16))
    assert np.isfinite(loss) and loss >= 0
    assert learner.updates == 1
    for key, value in learner.online.state_dict().items():
        assert torch.equal(value, learner.target.state_dict()[key])


def test_training_smoke() -> None:
    from examples.train_aec import run_training

    scores = run_training(episodes=2, horizon=3.0, n_iot=2, n_edge=2, batch_size=4, seed=11)
    assert len(scores) == 2
    assert all(np.isfinite(scores)) and all(score > 0 for score in scores)
