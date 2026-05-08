from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import mujoco

from physics_ai.config import TerminationConfig, default_joint_angles
from physics_ai.envs.h1_structs import EnvConfig
from physics_ai.utils.jax_utils import quat_rotate_inverse


def build_default_qpos(
    mj_model: mujoco.MjModel,
    override_joint_angles: dict[str, float] | None,
) -> jnp.ndarray:
    qpos = jnp.array(mj_model.qpos0).copy()
    joint_angles = default_joint_angles()
    if override_joint_angles:
        joint_angles.update(override_joint_angles)

    for joint_name, angle in joint_angles.items():
        joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            qpos_adr = mj_model.jnt_qposadr[joint_id]
            qpos = qpos.at[qpos_adr].set(angle)
    return qpos


def build_default_ctrl(
    mj_model: mujoco.MjModel,
    override_joint_angles: dict[str, float] | None,
) -> jnp.ndarray:
    joint_angles = default_joint_angles()
    if override_joint_angles:
        joint_angles.update(override_joint_angles)

    ctrl = []
    for actuator_idx in range(mj_model.nu):
        joint_id = mj_model.actuator_trnid[actuator_idx, 0]
        joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        ctrl.append(joint_angles.get(joint_name, 0.0))
    return jnp.array(ctrl)


def reset_qpos(default_qpos: jnp.ndarray, initial_height: float) -> jnp.ndarray:
    return default_qpos.at[2].set(initial_height)


def compute_observation(
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    joint_qpos_indices: jnp.ndarray,
    joint_qvel_indices: jnp.ndarray,
    command: jnp.ndarray,
) -> jnp.ndarray:
    joint_pos = qpos[..., joint_qpos_indices]
    joint_vel = qvel[..., joint_qvel_indices]
    base_quat = qpos[..., 3:7]
    base_ang_vel = qvel[..., 3:6]

    gravity_world = jnp.array([0.0, 0.0, -1.0])
    if qpos.ndim == 2:
        projected_gravity = jax.vmap(quat_rotate_inverse)(
            base_quat,
            jnp.tile(gravity_world, (qpos.shape[0], 1)),
        )
    else:
        projected_gravity = quat_rotate_inverse(base_quat, gravity_world)

    return jnp.concatenate(
        [
            joint_pos,
            joint_vel,
            base_quat,
            base_ang_vel,
            command,
            projected_gravity,
        ],
        axis=-1,
    )


def sample_commands_batched(rng: jax.Array, num_envs: int, config: EnvConfig) -> jnp.ndarray:
    rng, *keys = jax.random.split(rng, 4)
    vx = jax.random.uniform(keys[0], (num_envs,), minval=config.vx_range[0], maxval=config.vx_range[1])
    vy = jax.random.uniform(keys[1], (num_envs,), minval=config.vy_range[0], maxval=config.vy_range[1])
    vyaw = jax.random.uniform(keys[2], (num_envs,), minval=config.vyaw_range[0], maxval=config.vyaw_range[1])
    return jnp.stack([vx, vy, vyaw], axis=-1)


def sample_command_single(rng: jax.Array, config: EnvConfig) -> jnp.ndarray:
    rng, *keys = jax.random.split(rng, 4)
    vx = jax.random.uniform(keys[0], (), minval=config.vx_range[0], maxval=config.vx_range[1])
    vy = jax.random.uniform(keys[1], (), minval=config.vy_range[0], maxval=config.vy_range[1])
    vyaw = jax.random.uniform(keys[2], (), minval=config.vyaw_range[0], maxval=config.vyaw_range[1])
    return jnp.stack([vx, vy, vyaw])


def compute_termination(
    qpos: jnp.ndarray,
    termination_config: TerminationConfig,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    base_height = qpos[..., 2]
    base_quat = qpos[..., 3:7]
    gravity_world = jnp.array([0.0, 0.0, -1.0])

    if qpos.ndim == 2:
        projected_gravity = jax.vmap(quat_rotate_inverse)(
            base_quat,
            jnp.tile(gravity_world, (qpos.shape[0], 1)),
        )
    else:
        projected_gravity = quat_rotate_inverse(base_quat, gravity_world)

    pitch = jnp.arcsin(-projected_gravity[..., 0])
    roll = jnp.arctan2(projected_gravity[..., 1], projected_gravity[..., 2])

    fallen = base_height < termination_config.min_height
    tilted = (jnp.abs(pitch) > termination_config.max_pitch) | (
        jnp.abs(roll) > termination_config.max_roll
    )
    done = fallen | tilted

    return done, {
        "base_height": base_height,
        "pitch": pitch,
        "roll": roll,
        "projected_gravity": projected_gravity,
    }


def actuator_mode_summary(mj_model: mujoco.MjModel) -> dict[str, Any]:
    actuator_types: list[str] = []
    for actuator_type in mj_model.actuator_biastype:
        actuator_types.append(str(int(actuator_type)))

    is_position_control = bool(jnp.all(mj_model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED))
    return {
        "num_actuators": int(mj_model.nu),
        "ctrlrange": jnp.array(mj_model.actuator_ctrlrange).tolist(),
        "actuator_biastype": actuator_types,
        "position_like_control": is_position_control,
    }
