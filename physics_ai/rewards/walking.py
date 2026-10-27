from __future__ import annotations

from dataclasses import dataclass

import flax.struct
import jax.numpy as jnp


@dataclass
class RewardConfig:
    """Weights follow unitree_rl_gym H1 / MuJoCo Playground; signs are in the weights.

    Lesson from earlier runs: tracking_sigma must stay small (0.25). A forgiving
    sigma lets a robot standing still collect most of the velocity reward.
    """

    tracking_lin_vel: float = 1.0
    tracking_ang_vel: float = 0.5
    tracking_sigma: float = 0.25
    contact_match: float = 0.2
    feet_air_time: float = 1.0
    feet_air_time_max: float = 0.5
    feet_phase: float = 1.0
    feet_phase_sigma: float = 0.01
    swing_height_target: float = 0.08
    feet_slip: float = -0.05
    orientation: float = -1.0
    base_height: float = -10.0
    target_height: float = 0.98
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05
    action_rate: float = -0.01
    torques: float = -1e-5
    hip_deviation: float = -1.0
    alive: float = 0.15
    termination: float = -10.0
    move_command_threshold: float = 0.1


@flax.struct.dataclass
class RewardInputs:
    """Everything the reward needs for one env at one control step."""

    base_lin_vel: jnp.ndarray       # (3,) pelvis frame
    base_ang_vel: jnp.ndarray       # (3,) pelvis frame
    projected_gravity: jnp.ndarray  # (3,)
    base_height: jnp.ndarray        # ()
    command: jnp.ndarray            # (3,) vx, vy, yaw rate
    action: jnp.ndarray             # (nu,)
    prev_action: jnp.ndarray        # (nu,)
    torques: jnp.ndarray            # (nu,)
    hip_offset: jnp.ndarray         # (4,) hip yaw/roll minus default
    foot_contact: jnp.ndarray       # (2,) bool
    desired_contact: jnp.ndarray    # (2,) bool, from the gait clock
    first_contact: jnp.ndarray      # (2,) bool, touchdown this step
    air_time: jnp.ndarray           # (2,) seconds airborne before touchdown
    foot_height: jnp.ndarray        # (2,) sole height above floor
    foot_height_target: jnp.ndarray # (2,) gait-clock target sole height
    foot_vel_xy: jnp.ndarray        # (2, 2) world-frame horizontal foot velocity
    done: jnp.ndarray               # () fell this step (not truncation)


REWARD_TERMS = (
    "tracking_lin_vel",
    "tracking_ang_vel",
    "contact_match",
    "feet_air_time",
    "feet_phase",
    "feet_slip",
    "orientation",
    "base_height",
    "lin_vel_z",
    "ang_vel_xy",
    "action_rate",
    "torques",
    "hip_deviation",
    "alive",
    "termination",
)


def compute_reward_terms(inputs: RewardInputs, config: RewardConfig) -> dict[str, jnp.ndarray]:
    """Unweighted reward terms for a single env."""
    moving = jnp.linalg.norm(inputs.command[:2]) > config.move_command_threshold
    # Standing still is commanded: both feet should stay down.
    desired_contact = jnp.where(moving, inputs.desired_contact, True)
    foot_height_target = jnp.where(moving, inputs.foot_height_target, 0.0)

    lin_vel_error = jnp.sum((inputs.base_lin_vel[:2] - inputs.command[:2]) ** 2)
    ang_vel_error = (inputs.base_ang_vel[2] - inputs.command[2]) ** 2

    air_time = jnp.clip(inputs.air_time, 0.0, config.feet_air_time_max)
    foot_height_error = jnp.sum((inputs.foot_height - foot_height_target) ** 2)

    return {
        "tracking_lin_vel": jnp.exp(-lin_vel_error / config.tracking_sigma),
        "tracking_ang_vel": jnp.exp(-ang_vel_error / config.tracking_sigma),
        "contact_match": jnp.sum(inputs.foot_contact == desired_contact).astype(jnp.float32),
        # Never negative: shuffling earns little, but lifting a foot is never worse than not.
        "feet_air_time": jnp.sum(air_time * inputs.first_contact) * moving,
        # Dense stepping signal: feet follow the gait clock's height curve.
        "feet_phase": jnp.exp(-foot_height_error / config.feet_phase_sigma),
        "feet_slip": jnp.sum(jnp.sum(inputs.foot_vel_xy**2, axis=-1) * inputs.foot_contact),
        "orientation": jnp.sum(inputs.projected_gravity[:2] ** 2),
        "base_height": (inputs.base_height - config.target_height) ** 2,
        "lin_vel_z": inputs.base_lin_vel[2] ** 2,
        "ang_vel_xy": jnp.sum(inputs.base_ang_vel[:2] ** 2),
        "action_rate": jnp.sum((inputs.action - inputs.prev_action) ** 2),
        "torques": jnp.sum(inputs.torques**2),
        "hip_deviation": jnp.sum(inputs.hip_offset**2),
        "alive": jnp.ones(()),
        "termination": inputs.done.astype(jnp.float32),
    }


def compute_reward(
    inputs: RewardInputs, config: RewardConfig
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Total reward plus each weighted term, for logging."""
    terms = compute_reward_terms(inputs, config)
    weighted = {name: getattr(config, name) * terms[name] for name in REWARD_TERMS}
    total = sum(weighted.values())
    return total, weighted
