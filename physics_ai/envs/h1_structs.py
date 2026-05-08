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
    initial_height: float = 0.88
    vx_range: tuple[float, float] = (-1.0, 1.0)
    vy_range: tuple[float, float] = (-0.5, 0.5)
    vyaw_range: tuple[float, float] = (-1.0, 1.0)
    command_resample_time: float = 10.0
