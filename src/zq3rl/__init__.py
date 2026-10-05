"""ZQ3-Recovery-RL: predictive-perception + residual reinforcement learning
guidance for reusable launch-vehicle vertical recovery.

The package is deliberately dependency-light (NumPy + PyTorch + Matplotlib) so
that the whole experiment is reproducible on a CPU-only laptop.
"""

from .config import (
    Config,
    EnvConfig,
    EstimatorConfig,
    GuidanceConfig,
    MissionConfig,
    PPOConfig,
    RocketConfig,
    SensorConfig,
    WindConfig,
    air_density,
)

__version__ = "1.0.0"
__all__ = [
    "Config",
    "RocketConfig",
    "MissionConfig",
    "WindConfig",
    "SensorConfig",
    "EstimatorConfig",
    "GuidanceConfig",
    "EnvConfig",
    "PPOConfig",
    "air_density",
    "__version__",
]
