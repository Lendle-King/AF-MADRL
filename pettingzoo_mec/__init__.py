"""PettingZoo-compatible event-driven MEC benchmark.

The original paper implementation remains in the repository root.
"""
from .aec_env import AgeMinimalMECEnv, MECConfig, env


def register_env(env_id: str = "af-madrl/age-minimal-mec-v0") -> None:
    """Explicitly register this third-party AEC environment (PettingZoo >= 1.27)."""
    from pettingzoo import register

    register("aec", env_id, env)


__all__ = ["AgeMinimalMECEnv", "MECConfig", "env", "register_env"]
