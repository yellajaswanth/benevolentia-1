from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnvConfig:
    num_envs: int = 4096
    episode_length: int = 1000
    dt: float = 0.005
    control_decimation: int = 4
    action_scale: float = 0.25
    default_joint_angles: dict[str, float] | None = None
    # None: derived from the default pose so the feet start on the floor.
    initial_height: float | None = None
    vx_range: tuple[float, float] = (-1.0, 1.0)
    vy_range: tuple[float, float] = (-0.5, 0.5)
    vyaw_range: tuple[float, float] = (-1.0, 1.0)
    command_resample_time: float = 10.0
    # Gait clock frequency (Hz) = base + per_mps * commanded planar speed, so
    # slow walking takes slower, longer steps instead of a fixed fast cadence.
    gait_frequency_base: float = 0.75
    gait_frequency_per_mps: float = 0.5
    stance_ratio: float = 0.6
    # Fraction of sampled commands that are "stand still".
    stand_probability: float = 0.0
