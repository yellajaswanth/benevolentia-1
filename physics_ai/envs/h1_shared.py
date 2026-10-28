from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco

from physics_ai.config import TerminationConfig, default_joint_angles
from physics_ai.envs.h1_structs import EnvConfig
from physics_ai.utils.jax_utils import quat_rotate_inverse


# Observation layout version. Bump whenever compute_observation changes so stale
# checkpoints fail loudly instead of silently reading the wrong features.
OBS_VERSION = 2

FLOOR_GEOM_NAME = "floor"
FOOT_SITE_NAMES = ("left_foot", "right_foot")
FOOT_GEOM_NAMES = (
    ("left_foot1", "left_foot2", "left_foot3"),
    ("right_foot1", "right_foot2", "right_foot3"),
)
HIP_YAW_ROLL_JOINTS = ("left_hip_yaw", "left_hip_roll", "right_hip_yaw", "right_hip_roll")
TORSO_JOINT = "torso"
# Each arm swings with the opposite leg: (shoulder pitch, opposite hip pitch).
ARM_SWING_PAIRS = (
    ("left_shoulder_pitch", "right_hip_pitch"),
    ("right_shoulder_pitch", "left_hip_pitch"),
)
ARM_POSTURE_JOINTS = (
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
)


def default_asset_path() -> Path:
    # scene.xml includes h1.xml plus the floor. h1.xml alone has no ground plane.
    return Path(__file__).parent.parent.parent / "assets" / "unitree_h1" / "scene.xml"


def restrict_contacts_to_feet(mj_model: mujoco.MjModel) -> None:
    """Only feet collide with the floor; everything else is contact-free.

    Falls are caught by termination, so body-floor contacts add cost without
    adding signal. Mirrors the feet-only MJX scenes used for locomotion RL.
    """
    floor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)
    foot_ids = {
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for names in FOOT_GEOM_NAMES
        for name in names
    }
    for geom_id in range(mj_model.ngeom):
        if geom_id == floor_id:
            mj_model.geom_contype[geom_id] = 1
            mj_model.geom_conaffinity[geom_id] = 0
        elif geom_id in foot_ids:
            mj_model.geom_contype[geom_id] = 0
            mj_model.geom_conaffinity[geom_id] = 1
        else:
            mj_model.geom_contype[geom_id] = 0
            mj_model.geom_conaffinity[geom_id] = 0


def name_ids(mj_model: mujoco.MjModel, obj: mujoco.mjtObj, names: tuple[str, ...]) -> jnp.ndarray:
    ids = [mujoco.mj_name2id(mj_model, obj, name) for name in names]
    missing = [name for name, idx in zip(names, ids) if idx < 0]
    if missing:
        raise ValueError(f"Model is missing {obj}: {missing}")
    return jnp.array(ids)


def standing_height(mj_model: mujoco.MjModel, default_qpos: jnp.ndarray) -> float:
    """Pelvis height that puts the lower foot sole exactly on the floor."""
    data = mujoco.MjData(mj_model)
    data.qpos[:] = jax.device_get(default_qpos)
    data.qpos[2] = 0.0
    mujoco.mj_forward(mj_model, data)
    lowest_sole = min(data.site(name).xpos[2] for name in FOOT_SITE_NAMES)
    return float(-lowest_sole)


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


def base_kinematics(qpos: jnp.ndarray, qvel: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Base velocities and gravity in the pelvis frame (single env).

    Free-joint qvel[0:3] is world-frame linear velocity, but qvel[3:6] is
    already body-frame angular velocity, so only the linear part is rotated.
    """
    base_quat = qpos[3:7]
    return {
        "lin_vel": quat_rotate_inverse(base_quat, qvel[:3]),
        "ang_vel": qvel[3:6],
        "projected_gravity": quat_rotate_inverse(base_quat, jnp.array([0.0, 0.0, -1.0])),
    }


def gait_clock(phase: jnp.ndarray) -> jnp.ndarray:
    """sin/cos of each foot's phase; the right foot runs half a cycle behind."""
    foot_phase = jnp.stack([phase, (phase + 0.5) % 1.0])
    angle = 2.0 * jnp.pi * foot_phase
    return jnp.concatenate([jnp.sin(angle), jnp.cos(angle)])


def desired_foot_contact(phase: jnp.ndarray, stance_ratio: float) -> jnp.ndarray:
    foot_phase = jnp.stack([phase, (phase + 0.5) % 1.0])
    return foot_phase < stance_ratio


def desired_foot_height(phase: jnp.ndarray, stance_ratio: float, swing_height: float) -> jnp.ndarray:
    """(2,) target sole height: zero in stance, a sine bump peaking at swing_height mid-swing."""
    foot_phase = jnp.stack([phase, (phase + 0.5) % 1.0])
    swing_progress = jnp.clip((foot_phase - stance_ratio) / (1.0 - stance_ratio), 0.0, 1.0)
    return jnp.where(foot_phase >= stance_ratio, swing_height * jnp.sin(jnp.pi * swing_progress), 0.0)


def compute_observation(
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    joint_qpos_indices: jnp.ndarray,
    joint_qvel_indices: jnp.ndarray,
    default_joint_pos: jnp.ndarray,
    command: jnp.ndarray,
    phase: jnp.ndarray,
    prev_action: jnp.ndarray,
) -> jnp.ndarray:
    """Observation v2 (single env). Layout, in order:

    joint_pos - default (nj), joint_vel (nj), base_lin_vel (3), base_ang_vel (3),
    projected_gravity (3), command (3), gait_clock (4), prev_action (nu).

    base_lin_vel is privileged (not measured on hardware); acceptable for the
    simulation baseline.
    """
    base = base_kinematics(qpos, qvel)
    return jnp.concatenate(
        [
            qpos[joint_qpos_indices] - default_joint_pos,
            qvel[joint_qvel_indices],
            base["lin_vel"],
            base["ang_vel"],
            base["projected_gravity"],
            command,
            gait_clock(phase),
            prev_action,
        ]
    )


def observation_size(num_joints: int, num_actions: int) -> int:
    return num_joints * 2 + 3 + 3 + 3 + 3 + 4 + num_actions


FOOT_CONTACT_TOLERANCE = 1e-3  # metres; a sole resting exactly on the floor has dist == 0


def foot_contact(contact: Any, floor_id: jnp.ndarray, foot_geom_ids: jnp.ndarray) -> jnp.ndarray:
    """(2,) bool: whether any geom of each foot touches the floor."""
    touching_floor = (contact.geom1 == floor_id) | (contact.geom2 == floor_id)
    penetrating = touching_floor & (contact.dist < FOOT_CONTACT_TOLERANCE)

    def foot_touches(geom_ids: jnp.ndarray) -> jnp.ndarray:
        is_foot = jnp.isin(contact.geom1, geom_ids) | jnp.isin(contact.geom2, geom_ids)
        return jnp.any(penetrating & is_foot)

    return jax.vmap(foot_touches)(foot_geom_ids)


def sample_command_single(rng: jax.Array, config: EnvConfig) -> jnp.ndarray:
    rng, *keys = jax.random.split(rng, 5)
    vx = jax.random.uniform(keys[0], (), minval=config.vx_range[0], maxval=config.vx_range[1])
    vy = jax.random.uniform(keys[1], (), minval=config.vy_range[0], maxval=config.vy_range[1])
    vyaw = jax.random.uniform(keys[2], (), minval=config.vyaw_range[0], maxval=config.vyaw_range[1])
    stand = jax.random.uniform(keys[3]) < config.stand_probability
    return jnp.where(stand, 0.0, jnp.stack([vx, vy, vyaw]))


def gait_frequency(command: jnp.ndarray, config: EnvConfig) -> jnp.ndarray:
    return config.gait_frequency_base + config.gait_frequency_per_mps * jnp.linalg.norm(command[:2])


def feet_lateral_gap(qpos: jnp.ndarray, foot_pos: jnp.ndarray) -> jnp.ndarray:
    """Left-minus-right sole distance along the pelvis's lateral axis."""
    local = jax.vmap(lambda p: quat_rotate_inverse(qpos[3:7], p - qpos[:3]))(foot_pos)
    return local[0, 1] - local[1, 1]


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

    # Upright gravity in the pelvis frame is (0, 0, -1). Negate y/z so upright
    # gives roll = 0; arctan2(g_y, g_z) would give pi and terminate every step.
    pitch = jnp.arcsin(jnp.clip(-projected_gravity[..., 0], -1.0, 1.0))
    roll = jnp.arctan2(-projected_gravity[..., 1], -projected_gravity[..., 2])

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
