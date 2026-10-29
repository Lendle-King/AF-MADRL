# AF-MADRL

Source code for the paper **Asynchronous Fractional Multi-Agent Deep Reinforcement Learning for Age-Minimal Mobile Edge Computing**.

## PettingZoo AEC benchmark

This repository also contains an **independent, PettingZoo 1.27+ compatible
AoI-MEC benchmark**, designed for event-driven multi-agent reinforcement learning.
Each IoT agent makes offloading and waiting decisions at irregular event times;
edge servers use FCFS queues and the environment integrates age of information
over continuous time.

The environment can run **without PyTorch or CUDA**. From the repository root,
using Python 3.11 or 3.12:

```bash
python -m pip install -e ".[test]"
python -m pytest -q tests/test_pettingzoo_aec.py tests/test_event_dynamics.py tests/test_legacy_alignment.py
python examples/rollout_aec.py
```

Optional CPU-friendly Dueling Double DQN example:

```bash
python -m pip install -e ".[train,test]"
python -m pytest -q
python examples/train_aec.py --episodes 5 --horizon 20 --seed 42
```

A working PettingZoo 1.27 registry example, configuration and API descriptions,
reproducibility tests, and research limitations are available in
[PETTINGZOO.md](PETTINGZOO.md).

**Scope:** This new benchmark is **not an exact reproduction of the paper**.
The original simulator and DRQN/GRU-PPO code are preserved at the repository
root. The new benchmark does not yet implement the full channel model or
Dinkelbach/GRU-PPO training, and legacy and new AoI initialization conventions
differ. Do not compare their performance numbers as interchangeable.
