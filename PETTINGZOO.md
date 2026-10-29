# AF-MADRL: PettingZoo AEC benchmark (experimental branch)

This branch adds an **independent PettingZoo-compatible benchmark** based on
the asynchronous AoI/minimum-age mobile edge-computing setting of AF-MADRL.

**Important scientific boundary:** the original paper code (the files
mec_env_test.py, MA_DQN.py, RNN_PPO.py and main_MA_DQN_PPO.py) remains
untouched. The new benchmark captures core event-driven, edge-FCFS and AoI
features but **does not reproduce the paper's complete channel model,
fractional Dinkelbach update, initial task distribution or original GRU-PPO
schedule**. Do not claim reproduction of the paper's performance metrics
without a separate equivalence study.

## Install

Python 3.11 or 3.12:

~~~bash
python -m pip install -e ".[test]"
python -m pytest -q tests/test_pettingzoo_aec.py
python examples/rollout_aec.py
~~~

Training requires PyTorch (CPU is fine):

~~~bash
python -m pip install -e ".[train,test]"
python -m pytest -q
python examples/train_aec.py --episodes 5 --horizon 20 --seed 42
~~~

The root requirements.txt targets the historical paper implementation and
is deliberately not used by this new package.

## Environment API

~~~python
import numpy as np
from pettingzoo_mec import env, MECConfig

simulation = env(config=MECConfig(n_iot=4, n_edge=2, max_time=20))
simulation.reset(seed=42)
for agent in simulation.agent_iter():
    obs, reward, terminated, truncated, info = simulation.last()
    if terminated or truncated:
        action = None
    else:
        action = simulation.action_space(agent).sample()
        if info["phase"] == "wait":
            action["wait"] = np.array([1.0], dtype=np.float32)
    simulation.step(action)
print(simulation.mean_aoi)
simulation.close()
~~~

- The selected agent is the IoT device whose **next event** occurs first.
- Action is a fixed Gymnasium Dict with two fields:
  - offload: Discrete(n_edge + 1); 0=local, 1..n_edge=edge index.
  - wait: Box([0], [wait_max]); seconds until the next generated task.
  - Only the field matching info["phase"] is used. Keeping the same action
    space throughout the episode is intentional.
- Each agent's observation is a float32 vector:
  [edge_queue_lengths..., own_AoI, remaining_time, is_wait_phase,
   last_end_to_end_latency, current_time].
- Offloading creates a task whose processing duration is exponential with
  mean task_size * computation_density / processor_capacity. Every edge
  executes tasks FCFS. At completion the device chooses a waiting duration.
- Decision times are irregular. Each agent's reward is minus the **integral
  of AoI over real time** divided by reward_scale. PettingZoo AEC cumulative
  rewards aggregate between an agent's turns. mean_aoi is the time-average
  AoI across IoT devices.
- Reaching max_time sets truncations for all agents, followed by final
  step(None) calls to retire them properly.
- state() provides global observations for CTDE algorithms.
- Seeding uses a per-instance NumPy generator and seeded action spaces.

## Reference offloading learner

pettingzoo_mec/agents.py provides a small **Dueling Double DQN** baseline,
not a GRU-PPO reproduction. It addresses pitfalls found in static review:

1. Epsilon-greedy samples from all n_edge + 1 actions.
2. Both dueling value and advantage branches consume the learned features.
3. Bootstrapping is multiplied by (1 - done), not done.
4. Online network selects and target network evaluates the next action.
5. SMDP offloading transitions aggregate rewards until the same IoT's next
   offloading decision, including the intervening waiting phase.
6. Irregular-time discount is discount_factor ** (elapsed / reference_time);
   the Dinkelbach parameter used by the paper is a different variable.
7. Reproducible initialization, gradient clipping and periodic target sync.

The waiting subpolicy is **fixed**, not learned. This separates offloading
learning from continuous-action policy research until the environment is
tested.

## PettingZoo 1.27 registry

The new PettingZoo registry can be used after explicit registration:

~~~python
from pettingzoo import make
from pettingzoo_mec import register_env

register_env()
simulation = make("aec", "af-madrl/age-minimal-mec-v0")
~~~

## Verification and upstreaming

~~~bash
python -m pytest -q  # install .[train,test] first
python -m pytest -q tests/test_pettingzoo_aec.py tests/test_event_dynamics.py tests/test_legacy_alignment.py
python -m pytest -q tests/test_offloading_agent.py
~~~

The tests cover the official api_test, seed_test, seed reproducibility,
spaces, FCFS queue ordering, AoI integration, truncation, variable-time
discount, and CPU training smoke test.

PettingZoo generally recommends listing independent research environments
in the **third-party environments directory** rather than merging them into
the core library. Only propose that PR once compatibility and reproducibility
tests pass.

## Additional correctness constraints

- **No future latency disclosure:** the last observed end-to-end latency is
  updated only after a task is delivered. In-flight service samples are
  simulator-internal and never exposed in observations or CTDE state().
- **Time-limit bootstrap:** the SMDP DQN replay buffer distinguishes actual
  termination from PettingZoo time-limit truncation. Truncated trajectories
  keep the target-value bootstrap; rewards still include the complete
  accumulated AoI cost up to the horizon.
- **Exact integrated cost:** the environment exposes the *undiscounted*
  real-time AoI integral. The demonstration agent uses elapsed-time discount
  for future value. This is a practical episodic baseline, not an exact
  continuous-time discounted-reward solver: an exact discounted objective
  would also discount the reward rate inside each holding interval.
- **No legacy equivalence claim:** the original simulator's AoI averaging,
  task initialization and fractional reward conventions differ. Paper metric
  reproduction requires matching the legacy task process and comparing
  identical scripted trajectories and random samples.

## Legacy / AEC numerical alignment

A deterministic side-by-side regression test, tests/test_legacy_alignment.py,
runs the **actual legacy Offload class** and the AEC environment with identical
two-device, one-edge FCFS service durations of 2 and 1 time units.

- The first and second completion times are 2 and 3 in **both** environments,
  and their edge queue occupancies agree at these completion events.
- The legacy implementation initializes tasks automatically on reset and
  uses a queue-delay surrogate. The AEC environment starts with explicit
  per-agent offloading decisions and schedules service at its true FCFS start.
- Their reported AoI quantities do **not** agree at startup: legacy accumulates
  completed-update trapezoid terms, whereas the AEC benchmark continuously
  integrates age from t=0. For that scripted trajectory, the legacy aoi_average
  remains zero through the first update for each device, while AEC mean_aoi at
  t=3 equals 1.5. This is a **definition/initialization mismatch**, not evidence
  of identical paper benchmark behavior.
- Legacy fractional rewards also depend on the Dinkelbach gamma term and
  cannot be compared directly to the AEC's time-integral rewards.

Do not report results from the two systems as interchangeable. Matching the
paper needs an explicit mapping for warmup/boundary conditions, wait actions,
channels, reward convention and the original update schedule.

## Distribution smoke test

The GitHub Actions matrix covers Python 3.11 and 3.12, and on Python 3.11
builds both source and wheel distributions with `python -m build`, installs
the generated wheel, and instantiates the AEC environment. The `test` extra
includes SciPy solely for the legacy parity test; the normal environment
has no SciPy or PyTorch runtime dependency.

For upstream third-party listing, first ask the PettingZoo maintainers on
their linked Discord as requested in their documentation. The proposed
listing should reference only the **independent AEC benchmark**, not claim
that the legacy fractional-RL paper has been reproduced.
