from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx

from physics_ai.utils.jax_utils import quat_rotate_inverse


@dataclass
class RewardConfig:
    reward_scaling: float = 0.1
    velocity_tracking_weight: float = 1.0
    velocity_tracking_scale: float = 0.25
    yaw_rate_weight: float = 0.5
    yaw_rate_scale: float = 0.25
    upright_weight: float = 0.2
    height_weight: float = 0.1
    target_height: float = 0.98
    energy_weight: float = -0.001
    smoothness_weight: float = -0.01
    alive_bonus: float = 1.0
    termination_penalty: float = -10.0


def compute_reward(
    mjx_data: mjx.Data,
    action: jnp.ndarray,
    prev_action: jnp.ndarray,
    command: jnp.ndarray,
    joint_qpos_indices: jnp.ndarray,
    joint_qvel_indices: jnp.ndarray,
    config: RewardConfig | None = None,
) -> jnp.ndarray:
    if config is None:
        config = RewardConfig()
    
    qpos = mjx_data.qpos
    qvel = mjx_data.qvel
    
    base_pos = qpos[:, :3]
    base_quat = qpos[:, 3:7]
    base_lin_vel = qvel[:, :3]
    base_ang_vel = qvel[:, 3:6]
    
    base_lin_vel_local = jax.vmap(quat_rotate_inverse)(base_quat, base_lin_vel)
    base_ang_vel_local = jax.vmap(quat_rotate_inverse)(base_quat, base_ang_vel)
    
    vel_tracking = _velocity_tracking_reward(
        base_lin_vel_local[:, :2],
        command[:, :2],
        config.velocity_tracking_scale,
    )
    
    yaw_tracking = _yaw_rate_tracking_reward(
        base_ang_vel_local[:, 2],
        command[:, 2],
        config.yaw_rate_scale,
    )
    
    upright = _upright_reward(base_quat)
    
    height = _height_reward(base_pos[:, 2], config.target_height)
    
    energy = _energy_penalty(action)
    
    smoothness = _smoothness_penalty(action, prev_action)
    
    alive = jnp.ones(action.shape[0]) * config.alive_bonus
    
    total_reward = (
        config.velocity_tracking_weight * vel_tracking +
        config.yaw_rate_weight * yaw_tracking +
        config.upright_weight * upright +
        config.height_weight * height +
        config.energy_weight * energy +
        config.smoothness_weight * smoothness +
        alive
    )
    
    return total_reward * config.reward_scaling


def _velocity_tracking_reward(
    current_vel: jnp.ndarray,
    target_vel: jnp.ndarray,
    scale: float,
) -> jnp.ndarray:
    vel_error = jnp.sum((current_vel - target_vel) ** 2, axis=-1)
    return jnp.exp(-vel_error / scale)


def _yaw_rate_tracking_reward(
    current_yaw_rate: jnp.ndarray,
    target_yaw_rate: jnp.ndarray,
    scale: float,
) -> jnp.ndarray:
    yaw_error = (current_yaw_rate - target_yaw_rate) ** 2
    return jnp.exp(-yaw_error / scale)


def _upright_reward(base_quat: jnp.ndarray) -> jnp.ndarray:
    gravity_world = jnp.array([0.0, 0.0, -1.0])
    projected_gravity = jax.vmap(quat_rotate_inverse)(
        base_quat,
        jnp.tile(gravity_world, (base_quat.shape[0], 1)),
    )
    return projected_gravity[:, 2]


def _height_reward(
    current_height: jnp.ndarray,
    target_height: float,
) -> jnp.ndarray:
    height_error = jnp.abs(current_height - target_height)
    return jnp.exp(-height_error * 10.0)


def _energy_penalty(action: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(action ** 2, axis=-1)


def _smoothness_penalty(
    action: jnp.ndarray,
    prev_action: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.sum((action - prev_action) ** 2, axis=-1)


class WalkingReward:
    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()
    
    def __call__(
        self,
        mjx_data: mjx.Data,
        action: jnp.ndarray,
        prev_action: jnp.ndarray,
        command: jnp.ndarray,
        joint_qpos_indices: jnp.ndarray,
        joint_qvel_indices: jnp.ndarray,
    ) -> jnp.ndarray:
        return compute_reward(
            mjx_data=mjx_data,
            action=action,
            prev_action=prev_action,
            command=command,
            joint_qpos_indices=joint_qpos_indices,
            joint_qvel_indices=joint_qvel_indices,
            config=self.config,
        )

