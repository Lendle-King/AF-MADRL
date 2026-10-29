"""Reference SMDP-aware Dueling Double DQN for the offloading subpolicy.

This intentionally trains offload choices only. Waiting is managed separately
by a fixed policy in examples/train_aec.py. The two-stage design makes the
SMDP decision boundary explicit, rather than claiming the legacy GRU-PPO
and DRQN checkpoints are compatible with the new API.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def elapsed_discount(gamma: float, elapsed: float, reference_time: float = 1.0) -> float:
    """Time-consistent discount: gamma ** (elapsed / reference_time)."""
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if elapsed < 0 or reference_time <= 0:
        raise ValueError("elapsed must be nonnegative and reference_time positive")
    return gamma ** (elapsed / reference_time)


@dataclass(frozen=True)
class TimedTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    elapsed: float


class ReplayBuffer:
    def __init__(self, capacity: int = 20000, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._items: deque[TimedTransition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def add(self, transition: TimedTransition) -> None:
        if not np.isfinite(transition.reward) or transition.elapsed < 0:
            raise ValueError("transition reward or elapsed time is invalid")
        self._items.append(transition)

    def sample(self, batch_size: int) -> list[TimedTransition]:
        return self._rng.sample(list(self._items), batch_size)

    def __len__(self) -> int:
        return len(self._items)


class DuelingQNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        if observation_dim < 1 or action_dim < 2:
            raise ValueError("invalid observation or action dimension")
        self.backbone = nn.Sequential(
            nn.Linear(observation_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        feature = self.backbone(observations)
        value = self.value(feature)
        advantage = self.advantage(feature)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class DoubleDQNAgent:
    """Shared offloading policy; each IoT contributes timed transitions."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        gamma: float = 0.99,
        reference_time: float = 1.0,
        lr: float = 3e-4,
        target_sync: int = 100,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        elapsed_discount(gamma, 0.0, reference_time)
        if target_sync < 1:
            raise ValueError("target_sync must be positive")
        self.action_dim = action_dim
        self.gamma = gamma
        self.reference_time = reference_time
        self.target_sync = target_sync
        self.device = torch.device(device)
        self._rng = np.random.default_rng(seed)
        self.online = DuelingQNetwork(observation_dim, action_dim).to(self.device)
        self.target = DuelingQNetwork(observation_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.updates = 0

    def act(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        if self._rng.random() < epsilon:
            return int(self._rng.integers(0, self.action_dim))
        with torch.no_grad():
            x = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            return int(self.online(x.unsqueeze(0)).argmax(dim=-1).item())

    def learn(self, transitions: list[TimedTransition]) -> float:
        if not transitions:
            raise ValueError("empty batch")
        states = torch.as_tensor(
            np.stack([x.state for x in transitions]), dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            np.stack([x.next_state for x in transitions]), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor([x.action for x in transitions], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([x.reward for x in transitions], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([x.done for x in transitions], dtype=torch.float32, device=self.device)
        discounts = torch.as_tensor(
            [elapsed_discount(self.gamma, x.elapsed, self.reference_time) for x in transitions],
            dtype=torch.float32,
            device=self.device,
        )

        q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: the online network selects the next action and the
            # frozen target network evaluates it. Terminal transitions do not bootstrap.
            best_next = self.online(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, best_next).squeeze(1)
            td_target = rewards + (1.0 - dones) * discounts * next_q

        loss = F.smooth_l1_loss(q, td_target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().cpu().item())
