# AF-MADRL
Code for paper " Asynchronous Fractional Multi-Agent Deep  Reinforcement Learning for  Age-Minimal Mobile Edge Computing".

## PettingZoo-compatible benchmark (new, experimental)

The new [PettingZoo AEC integration](PETTINGZOO.md) provides an independent
asynchronous AoI-MEC environment and a reference SMDP-aware Dueling Double
DQN offloading baseline. Install with `pip install -e '.[train,test]'` and run
`python -m pytest -q` or `python examples/train_aec.py --episodes 5`.

This does **not** alter the original AF-MADRL paper implementation or claim
numerical equivalence to the original paper's experimental results.
