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


def test_exact_double_dqn_target_and_terminal_mask() -> None:
    """Checks the learning loss against a manually evaluated SMDP TD target."""
    learner = DoubleDQNAgent(7, 3, gamma=0.9, target_sync=100)
    with torch.no_grad():
        for network in (learner.online, learner.target):
            for p in network.parameters():
                p.zero_()
        learner.online.value.bias.fill_(1.0)
        learner.target.value.bias.fill_(3.0)

    obs = np.zeros(7, dtype=np.float32)
    one_terminal = TimedTransition(obs, 1, 2.0, obs, True, 2.0)
    one_continuing = TimedTransition(obs, 2, 2.0, obs, False, 2.0)
    measured = learner.learn([one_terminal, one_continuing])

    # q(s,a)=1, target Q(s',argmax Q)=3, elapsed gamma^2=0.81.
    # Terminal: y=2.0; nonterminal: y=2.0+0.81*3=4.43.
    # Smooth L1 beta=1: 0.5*(1)^2 and abs(4.43-1)-0.5.
    expected = (0.5 + (abs(2.0 + 0.81 * 3.0 - 1.0) - 0.5)) / 2
    assert measured == pytest.approx(expected, rel=1e-6)


def test_aec_training_reward_ledger_and_truncation_bootstrap(monkeypatch) -> None:
    """The offload replay must retain all AoI costs and not mask time limits."""
    from examples import train_aec

    collected: list[TimedTransition] = []
    original_add = train_aec.ReplayBuffer.add

    def collecting_add(self, transition):
        collected.append(transition)
        return original_add(self, transition)

    monkeypatch.setattr(train_aec.ReplayBuffer, "add", collecting_add)
    horizon, n_iot, scale = 2.0, 2, 100.0
    aoi = train_aec.run_training(
        episodes=1, horizon=horizon, n_iot=n_iot, n_edge=1,
        fixed_wait=0.5, batch_size=10000, seed=21,
    )[0]
    assert collected
    assert all(x.elapsed >= 0 for x in collected)
    assert all(x.reward <= 0 for x in collected)
    assert all(not x.done for x in collected)  # Only truncations, no true terminations.
    assert any(x.elapsed > 0 for x in collected)

    # Every IoT AoI cost is assigned to exactly one SMDP interval.
    total_replay_reward = sum(t.reward for t in collected)
    expected_total = -(aoi * n_iot * horizon / scale)
    assert total_replay_reward == pytest.approx(expected_total, rel=1e-8, abs=1e-8)


def test_cpu_training_reproducible_for_same_seed() -> None:
    from examples.train_aec import run_training

    params = dict(
        episodes=2, horizon=2.0, n_iot=2, n_edge=2,
        fixed_wait=0.5, batch_size=8, seed=17,
    )
    a = run_training(**params)
    b = run_training(**params)
    assert a == pytest.approx(b, rel=1e-9)
