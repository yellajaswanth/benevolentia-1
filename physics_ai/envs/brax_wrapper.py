from __future__ import annotations
import dataclasses
from typing import Any, Dict

import flax.struct
import jax
import jax.numpy as jnp
from brax.envs import base as brax_base
import mujoco
from mujoco import mjx

from physics_ai.config import TerminationConfig
from physics_ai.envs.domain_rand import DomainRandomizationConfig, any_enabled
from physics_ai.envs.h1_shared import (
    FLOOR_GEOM_NAME,
    FOOT_GEOM_NAMES,
    ARM_POSTURE_JOINTS,
    ARM_SWING_PAIRS,
    FOOT_SITE_NAMES,
    HIP_YAW_ROLL_JOINTS,
    TORSO_JOINT,
    OBS_VERSION,
    actuator_mode_summary,
    base_kinematics,
    build_default_ctrl,
    build_default_qpos,
    compute_observation,
    compute_termination,
    default_asset_path,
    desired_foot_contact,
    desired_foot_height,
    feet_lateral_gap,
    gait_frequency,
    foot_contact,
    name_ids,
    observation_size,
    reset_qpos,
    restrict_contacts_to_feet,
    sample_command_single,
    standing_height,
)
from physics_ai.envs.h1_structs import EnvConfig
from physics_ai.rewards.walking import REWARD_TERMS, RewardConfig, RewardInputs, compute_reward


@flax.struct.dataclass
class BraxState(brax_base.State):
    pipeline_state: mjx.Data
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    metrics: Dict[str, jnp.ndarray] = flax.struct.field(default_factory=dict)
    info: Dict[str, Any] = flax.struct.field(default_factory=dict)


class BraxH1EnvWrapper(brax_base.Env):
    def __init__(
        self,
        env_config: EnvConfig | None = None,
        reward_config: RewardConfig | None = None,
        dr_config: DomainRandomizationConfig | None = None,
        asset_path: str | None = None,
        termination_config: TerminationConfig | None = None,
    ):
        self._env_config = env_config or EnvConfig()
        self._reward_config = reward_config or RewardConfig()
        self._dr_config = dr_config
        self._termination_config = termination_config or TerminationConfig()

        # Domain randomization is not applied by this env yet (Phase 5). Refuse to
        # run with it enabled rather than silently training without it.
        if dr_config is not None and any_enabled(dr_config):
            raise NotImplementedError(
                "Domain randomization is enabled in the config but not implemented "
                "in BraxH1EnvWrapper. Disable all domain_randomization entries."
            )

        if asset_path is None:
            asset_path = default_asset_path()

        self._mj_model = mujoco.MjModel.from_xml_path(str(asset_path))
        self._mj_model.opt.timestep = self._env_config.dt
        restrict_contacts_to_feet(self._mj_model)
        self._mjx_model = mjx.put_model(self._mj_model)

        self._setup_indices()
        self._default_qpos = build_default_qpos(self._mj_model, self._env_config.default_joint_angles)
        self._default_ctrl = build_default_ctrl(self._mj_model, self._env_config.default_joint_angles)
        self._default_joint_pos = self._default_qpos[self._joint_qpos_indices]
        self._ctrl_min = jnp.array(self._mj_model.actuator_ctrlrange[:, 0])
        self._ctrl_max = jnp.array(self._mj_model.actuator_ctrlrange[:, 1])

        if self._env_config.initial_height is None:
            self._initial_height = standing_height(self._mj_model, self._default_qpos)
        else:
            self._initial_height = self._env_config.initial_height
        if self._reward_config.target_height is None:
            self._reward_config = dataclasses.replace(
                self._reward_config, target_height=self._initial_height
            )

    def _setup_indices(self) -> None:
        qpos_indices = []
        qvel_indices = []
        joint_names = []
        for i in range(self._mj_model.njnt):
            if self._mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
                qpos_indices.append(self._mj_model.jnt_qposadr[i])
                qvel_indices.append(self._mj_model.jnt_dofadr[i])
                joint_names.append(mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, i))
        self._joint_qpos_indices = jnp.array(qpos_indices)
        self._joint_qvel_indices = jnp.array(qvel_indices)

        self._floor_id = name_ids(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, (FLOOR_GEOM_NAME,))[0]
        self._foot_geom_ids = jnp.stack(
            [name_ids(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, names) for names in FOOT_GEOM_NAMES]
        )
        self._foot_site_ids = name_ids(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, FOOT_SITE_NAMES)
        index = lambda names: jnp.array([joint_names.index(name) for name in names])
        self._hip_joint_indices = index(HIP_YAW_ROLL_JOINTS)
        self._torso_joint_index = joint_names.index(TORSO_JOINT)
        self._shoulder_pitch_indices = index(tuple(arm for arm, _ in ARM_SWING_PAIRS))
        self._opposite_hip_pitch_indices = index(tuple(hip for _, hip in ARM_SWING_PAIRS))
        self._arm_posture_indices = index(ARM_POSTURE_JOINTS)

        self._num_actions = self._mj_model.nu
        self._obs_dim = observation_size(len(qpos_indices), self._num_actions)

    @property
    def observation_size(self) -> int:
        return self._obs_dim

    @property
    def action_size(self) -> int:
        return self._num_actions

    @property
    def backend(self) -> str:
        return "mjx"

    @property
    def obs_version(self) -> int:
        return OBS_VERSION

    @property
    def control_dt(self) -> float:
        return self._env_config.dt * self._env_config.control_decimation

    def reset(self, rng: jax.Array) -> BraxState:
        rng, cmd_key, phase_key = jax.random.split(rng, 3)

        data = mjx.make_data(self._mjx_model)
        qpos = reset_qpos(self._default_qpos, self._initial_height)
        data = data.replace(qpos=qpos, qvel=jnp.zeros_like(data.qvel), ctrl=self._default_ctrl)
        data = mjx.forward(self._mjx_model, data)

        command = self._sample_command(cmd_key)
        phase = jax.random.uniform(phase_key)
        prev_action = jnp.zeros(self._num_actions)

        metrics = {
            "episode_reward": jnp.array(0.0),
            "episode_length": jnp.array(0.0),
            **{f"reward/{name}": jnp.array(0.0) for name in REWARD_TERMS},
        }

        info = {
            "command": command,
            "prev_action": prev_action,
            "step_count": jnp.array(0, dtype=jnp.int32),
            "rng": rng,
            "truncation": jnp.array(0.0),
            "phase": phase,
            "air_time": jnp.zeros(2),
            "last_contact": jnp.ones(2, dtype=bool),
            "prev_foot_pos": data.site_xpos[self._foot_site_ids],
            # Brax's AutoResetWrapper restores only pipeline_state and obs from the
            # first reset; step() uses these to restore the rest of info to match.
            "first_command": command,
            "first_phase": phase,
            "first_foot_pos": data.site_xpos[self._foot_site_ids],
        }

        return BraxState(
            pipeline_state=data,
            obs=self._compute_obs(data, command, phase, prev_action),
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
            metrics=metrics,
            info=info,
        )

    def step(self, state: BraxState, action: jnp.ndarray) -> BraxState:
        rng, cmd_key = jax.random.split(state.info["rng"], 2)
        dt = self.control_dt

        ctrl = jnp.clip(
            self._default_ctrl + action * self._env_config.action_scale,
            self._ctrl_min,
            self._ctrl_max,
        )

        def physics_step(data, _):
            data = data.replace(ctrl=ctrl)
            return mjx.step(self._mjx_model, data), None

        mjx_data, _ = jax.lax.scan(
            physics_step,
            state.pipeline_state,
            None,
            length=self._env_config.control_decimation,
        )

        # Feet: contact, touchdown and air time (contact filtered over two steps
        # because MJX contacts can flicker for one step).
        contact = foot_contact(mjx_data.contact, self._floor_id, self._foot_geom_ids)
        contact_filt = contact | state.info["last_contact"]
        air_time = state.info["air_time"] + dt
        first_contact = (state.info["air_time"] > 0.0) & contact_filt
        foot_pos = mjx_data.site_xpos[self._foot_site_ids]
        foot_vel_xy = (foot_pos - state.info["prev_foot_pos"])[:, :2] / dt

        phase = state.info["phase"]
        done = self._compute_termination(mjx_data)
        base = base_kinematics(mjx_data.qpos, mjx_data.qvel)
        joint_pos = mjx_data.qpos[self._joint_qpos_indices]
        joint_offset = joint_pos - self._default_joint_pos
        arm_swing_target = self._reward_config.arm_swing_gain * joint_offset[self._opposite_hip_pitch_indices]

        reward, weighted_terms = compute_reward(
            RewardInputs(
                base_lin_vel=base["lin_vel"],
                base_ang_vel=base["ang_vel"],
                projected_gravity=base["projected_gravity"],
                base_height=mjx_data.qpos[2],
                command=state.info["command"],
                action=action,
                prev_action=state.info["prev_action"],
                torques=mjx_data.actuator_force,
                hip_offset=joint_offset[self._hip_joint_indices],
                torso_offset=joint_offset[self._torso_joint_index],
                arm_swing_error=joint_offset[self._shoulder_pitch_indices] - arm_swing_target,
                arm_posture_offset=joint_offset[self._arm_posture_indices],
                feet_lateral_gap=feet_lateral_gap(mjx_data.qpos, foot_pos),
                foot_contact=contact,
                desired_contact=desired_foot_contact(phase, self._env_config.stance_ratio),
                foot_height_target=desired_foot_height(
                    phase, self._env_config.stance_ratio, self._reward_config.swing_height_target
                ),
                first_contact=first_contact,
                air_time=air_time,
                foot_height=foot_pos[:, 2],
                foot_vel_xy=foot_vel_xy,
                done=done,
            ),
            self._reward_config,
        )

        # The env applies its own time limit (same episode_length as Brax's
        # EpisodeWrapper) so it sees every episode end and can reset its info.
        step_count = state.info["step_count"] + 1
        truncated = step_count >= self._env_config.episode_length
        done = done | truncated

        resample_interval = int(self._env_config.command_resample_time / dt)
        should_resample = (step_count % resample_interval) == 0
        new_command = jnp.where(
            should_resample,
            self._sample_command(cmd_key),
            state.info["command"],
        )
        new_phase = (phase + dt * gait_frequency(state.info["command"], self._env_config)) % 1.0

        metrics = {
            **state.metrics,
            **{f"reward/{name}": value for name, value in weighted_terms.items()},
        }

        info = {
            **state.info,
            "command": new_command,
            "prev_action": action,
            "step_count": step_count,
            "rng": rng,
            "truncation": jnp.where(truncated, 1.0, 0.0),
            "phase": new_phase,
            "air_time": air_time * ~contact_filt,
            "last_contact": contact,
            "prev_foot_pos": foot_pos,
        }
        info = self._reset_episode_info_where(done, info)

        return BraxState(
            pipeline_state=mjx_data,
            obs=self._compute_obs(mjx_data, new_command, new_phase, action),
            reward=reward,
            done=jnp.where(done, 1.0, 0.0),
            metrics=metrics,
            info=info,
        )

    def _reset_episode_info_where(self, done: jnp.ndarray, info: dict[str, Any]) -> dict[str, Any]:
        """Return per-episode info to its first-reset values where the episode ended.

        Without this, step_count, air time and the rest carried over across
        auto-resets (only pipeline_state and obs are restored by Brax).
        """
        fresh = {
            "command": info["first_command"],
            "prev_action": jnp.zeros_like(info["prev_action"]),
            "step_count": jnp.zeros_like(info["step_count"]),
            "phase": info["first_phase"],
            "air_time": jnp.zeros_like(info["air_time"]),
            "last_contact": jnp.ones_like(info["last_contact"]),
            "prev_foot_pos": info["first_foot_pos"],
        }
        return {
            **info,
            **{key: jnp.where(done, value, info[key]) for key, value in fresh.items()},
        }

    def _compute_obs(
        self,
        mjx_data: mjx.Data,
        command: jnp.ndarray,
        phase: jnp.ndarray,
        prev_action: jnp.ndarray,
    ) -> jnp.ndarray:
        return compute_observation(
            mjx_data.qpos,
            mjx_data.qvel,
            self._joint_qpos_indices,
            self._joint_qvel_indices,
            self._default_joint_pos,
            command,
            phase,
            prev_action,
        )

    def _sample_command(self, rng: jax.Array) -> jnp.ndarray:
        return sample_command_single(rng, self._env_config)

    def _compute_termination(self, mjx_data: mjx.Data) -> jnp.ndarray:
        done, _ = compute_termination(mjx_data.qpos, self._termination_config)
        return done

    def foot_contact(self, mjx_data: mjx.Data) -> jnp.ndarray:
        return foot_contact(mjx_data.contact, self._floor_id, self._foot_geom_ids)

    def set_command(self, state: BraxState, command: jnp.ndarray) -> BraxState:
        obs = self._compute_obs(
            state.pipeline_state, command, state.info["phase"], state.info["prev_action"]
        )
        info = {**state.info, "command": command}
        return state.replace(obs=obs, info=info)

    def diagnostics_summary(self) -> dict[str, Any]:
        return {
            "actuators": actuator_mode_summary(self._mj_model),
            "default_ctrl": jnp.array(self._default_ctrl).tolist(),
            "initial_height": self._initial_height,
            "obs_version": OBS_VERSION,
        }


def create_brax_h1_env(
    env_config: EnvConfig | None = None,
    reward_config: RewardConfig | None = None,
    dr_config: DomainRandomizationConfig | None = None,
    termination_config: TerminationConfig | None = None,
) -> BraxH1EnvWrapper:
    return BraxH1EnvWrapper(
        env_config=env_config,
        reward_config=reward_config,
        dr_config=dr_config,
        termination_config=termination_config,
    )
