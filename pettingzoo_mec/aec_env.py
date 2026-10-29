"""Event-driven AoI/MEC benchmark with the PettingZoo AEC interface.

This is a clean benchmark built from the assumptions of the AF-MADRL research
simulator, NOT a bit-for-bit wrapper around its original training experiments.
Agents alternate offloading and waiting decisions at asynchronous event times.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv


@dataclass(frozen=True)
class MECConfig:
    n_iot: int = 4
    n_edge: int = 2
    max_time: float = 40.0
    task_size: float = 8.0
    computation_density: float = 0.3
    local_capacity: float = 2.5
    edge_capacity: float = 41.8
    wait_max: float = 3.0
    reward_scale: float = 100.0

    def __post_init__(self) -> None:
        if self.n_iot < 1 or self.n_edge < 1:
            raise ValueError("n_iot and n_edge must be positive")
        if any(
            (not np.isfinite(x)) or x <= 0
            for x in (
                self.max_time, self.task_size, self.computation_density,
                self.local_capacity, self.edge_capacity, self.wait_max,
                self.reward_scale,
            )
        ):
            raise ValueError("all timing, capacity and scaling values must be positive")


class AgeMinimalMECEnv(AECEnv):
    """An asynchronous SMDP represented as a standard PettingZoo AEC environment.

    Each IoT agent alternates between an OFFLOAD event and a WAIT event:
      - OFFLOAD: action["offload"] selects local (0) or an edge (1..n_edge).
      - WAIT: action["wait"][0] selects delay before the next generated task.
    The inactive action field is ignored. Both fields are always present so the
    action space remains unchanged over the entire episode.

    The simulator holds at most one outstanding task per IoT. Edge servers use
    first-come-first-served queues. Processing durations are exponentially
    distributed with mean task_size * computation_density / capacity.

    AoI_i(t) = t - generation_time_of_last_delivered_update_i. The reward
    accrued for each agent across an event interval is minus the *exact*
    integral of its AoI divided by reward_scale. It is accumulated by AECEnv
    until that same agent acts again, even as other agents take turns.

    A fixed wall-clock horizon ends the episode via truncation, not termination.
    """

    metadata = {
        "name": "age_minimal_mec_v0",
        "render_modes": ["human", "ansi"],
        "is_parallelizable": False,
    }

    def __init__(
        self, config: MECConfig | None = None, render_mode: Literal["human", "ansi"] | None = None
    ) -> None:
        super().__init__()
        self.config = MECConfig() if config is None else config
        if render_mode not in (None, "human", "ansi"):
            raise ValueError("render_mode must be None, 'human' or 'ansi'")
        self.render_mode = render_mode
        self.possible_agents = [f"iot_{i}" for i in range(self.config.n_iot)]
        self._names = {a: i for i, a in enumerate(self.possible_agents)}
        self._observation_spaces = {
            a: spaces.Box(low=0, high=np.finfo(np.float32).max, shape=(self.config.n_edge + 5,), dtype=np.float32)
            for a in self.possible_agents
        }
        self._action_spaces = {
            a: spaces.Dict(
                {
                    "offload": spaces.Discrete(self.config.n_edge + 1),
                    "wait": spaces.Box(
                        low=np.array([0.0], dtype=np.float32),
                        high=np.array([self.config.wait_max], dtype=np.float32),
                        dtype=np.float32,
                    ),
                }
            )
            for a in self.possible_agents
        }
        self.reset()

    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_spaces[agent]

    @property
    def time(self) -> float:
        return self._time

    @property
    def mean_aoi(self) -> float:
        if self._time <= 0:
            return 0.0
        return float(self._aoi_areas.sum() / (self._time * self.config.n_iot))

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        del options
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            for i, agent in enumerate(self.possible_agents):
                self.action_space(agent).seed(seed + i)

        self.agents = self.possible_agents.copy()
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._time = 0.0
        self._aoi_areas = np.zeros(self.config.n_iot, dtype=np.float64)
        self._last_delivered_generation = np.zeros(self.config.n_iot, dtype=np.float64)
        self._last_latency = np.zeros(self.config.n_iot, dtype=np.float64)
        self._phase = ["offload"] * self.config.n_iot
        self._edge_available_until = np.zeros(self.config.n_edge, dtype=np.float64)
        self._edge_completions: list[list[float]] = [[] for _ in range(self.config.n_edge)]
        self._pending: list[tuple[float, int, str, int, float]] = []
        self._event_order = 0
        self._skip_agent_selection = None
        self.agent_selection = self.agents[0]
        for i in range(self.config.n_iot):
            self._schedule(0.0, "offload", i)
        self._advance_to_next_decision()
        self._update_infos(0.0)

    def observe(self, agent: str) -> np.ndarray:
        i = self._names[agent]
        age = max(0.0, self._time - self._last_delivered_generation[i])
        queue_lengths = [
            sum(finish > self._time + 1e-10 for finish in finishes)
            for finishes in self._edge_completions
        ]
        return np.asarray(
            [
                *queue_lengths,
                age,
                max(0.0, self.config.max_time - self._time),
                float(self._phase[i] == "wait"),
                self._last_latency[i],
                self._time,
            ],
            dtype=np.float32,
        )

    def state(self) -> np.ndarray:
        """Centralized (non-private) observations for CTDE algorithms."""
        return np.concatenate([self.observe(a) for a in self.possible_agents])

    def _schedule(
        self, timestamp: float, kind: Literal["offload", "wait", "complete"],
        i: int, generation_time: float = 0.0
    ) -> None:
        heapq.heappush(
            self._pending, (float(timestamp), self._event_order, kind, i, generation_time)
        )
        self._event_order += 1

    def _advance_clock(self, timestamp: float) -> None:
        delta_t = max(0.0, timestamp - self._time)
        if delta_t:
            age = self._time - self._last_delivered_generation
            area = age * delta_t + 0.5 * delta_t * delta_t
            self._aoi_areas += area
            for i, agent in enumerate(self.possible_agents):
                self.rewards[agent] -= float(area[i] / self.config.reward_scale)
            self._time = timestamp
        for edge in range(self.config.n_edge):
            self._edge_completions[edge] = [
                end for end in self._edge_completions[edge] if end > self._time + 1e-10
            ]

    def _advance_to_next_decision(self) -> None:
        while self._pending:
            timestamp, _, kind, i, generation_time = heapq.heappop(self._pending)
            if timestamp >= self.config.max_time:
                self._advance_clock(self.config.max_time)
                self.truncations = {a: True for a in self.agents}
                self._deads_step_first()
                return

            self._advance_clock(timestamp)
            if kind == "complete":
                # Completed updates reset AoI to their end-to-end service age.
                self._last_delivered_generation[i] = generation_time
                # The realized end-to-end latency is only known at delivery.
                # Publishing it during dispatch leaks future samples via state().
                self._last_latency[i] = self._time - generation_time
                self._schedule(self._time, "wait", i)
                continue

            self._phase[i] = kind
            self.agent_selection = self.possible_agents[i]
            return

        # Defensive: all pending events exhausted (should not happen for live agents).
        self.truncations = {a: True for a in self.agents}
        self._deads_step_first()

    def _update_infos(self, elapsed_time: float) -> None:
        for a in self.agents:
            i = self._names[a]
            self.infos[a] = {
                "time": self._time,
                "elapsed_time": float(elapsed_time),
                "phase": self._phase[i],
                "instantaneous_aoi": float(self._time - self._last_delivered_generation[i]),
                "mean_aoi": self.mean_aoi,
            }

    def step(self, action: dict[str, Any] | None) -> None:
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        if action is None or not self.action_space(self.agent_selection).contains(action):
            raise ValueError(f"invalid action for {self.agent_selection}: {action!r}")

        agent = self.agent_selection
        i = self._names[agent]
        self._cumulative_rewards[agent] = 0.0
        self._clear_rewards()
        previous_time = self._time

        if self._phase[i] == "offload":
            offload = int(action["offload"])
            if offload == 0:
                mean_duration = (
                    self.config.task_size * self.config.computation_density
                    / self.config.local_capacity
                )
                start = self._time
            else:
                edge = offload - 1
                mean_duration = (
                    self.config.task_size * self.config.computation_density
                    / self.config.edge_capacity
                )
                start = max(self._time, self._edge_available_until[edge])

            duration = max(1e-7, float(self._rng.exponential(mean_duration)))
            finish = start + duration
            if offload:
                self._edge_available_until[edge] = finish
                self._edge_completions[edge].append(finish)
            self._schedule(finish, "complete", i, generation_time=self._time)
        else:
            delay = float(action["wait"][0])
            self._schedule(self._time + delay, "offload", i)

        self._advance_to_next_decision()
        self._update_infos(self._time - previous_time)
        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()

    def render(self) -> str | None:
        if self.render_mode is None:
            return None
        line = (
            f"t={self._time:.3f}/{self.config.max_time:.3f} "
            f"mean_aoi={self.mean_aoi:.3f} next={self.agent_selection}"
        )
        if self.render_mode == "ansi":
            return line
        print(line)
        return None

    def close(self) -> None:
        return None


def env(**kwargs: Any) -> AgeMinimalMECEnv:
    """PettingZoo entry point: env(config=MECConfig(...), render_mode='ansi')."""
    return AgeMinimalMECEnv(**kwargs)
