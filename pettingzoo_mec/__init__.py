"""PettingZoo-compatible event-driven MEC benchmark.

The original paper implementation remains in the repository root.
"""
from .aec_env import AgeMinimalMECEnv, MECConfig, env

__all__ = ["AgeMinimalMECEnv", "MECConfig", "env"]
