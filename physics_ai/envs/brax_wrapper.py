from __future__ import annotations
from typing import Any, Dict

import flax.struct
import jax
import jax.numpy as jnp
from brax.envs import base as brax_base
import mujoco
from mujoco import mjx

from physics_ai.config import TerminationConfig
from physics_ai.envs.domain_rand import DomainRandomizationConfig
from physics_ai.envs.h1_shared import (
    actuator_mode_summary,
    build_default_ctrl,
    build_default_qpos,
    compute_observation,
    compute_termination,
    default_asset_path,
    reset_qpos,
    sample_command_single,
)
from physics_ai.envs.h1_structs import EnvConfig
from physics_ai.rewards.walking import RewardConfig
from physics_ai.rewards.walking import compute_reward_components


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
        
        if asset_path is None:
            asset_path = default_asset_path()
        
        self._mj_model = mujoco.MjModel.from_xml_path(str(asset_path))
        self._mj_model.opt.timestep = self._env_config.dt
        self._mjx_model = mjx.put_model(self._mj_model)
        
        self._setup_indices()
        self._default_qpos = build_default_qpos(self._mj_model, self._env_config.default_joint_angles)
        self._default_ctrl = build_default_ctrl(self._mj_model, self._env_config.default_joint_angles)
        self._ctrl_min = jnp.array(self._mj_model.actuator_ctrlrange[:, 0])
        self._ctrl_max = jnp.array(self._mj_model.actuator_ctrlrange[:, 1])

    def _setup_indices(self) -> None:
        self._joint_qpos_indices = []
        self._joint_qvel_indices = []
        
        for i in range(self._mj_model.njnt):
            jnt_type = self._mj_model.jnt_type[i]
            if jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
                qpos_adr = self._mj_model.jnt_qposadr[i]
                qvel_adr = self._mj_model.jnt_dofadr[i]
                self._joint_qpos_indices.append(qpos_adr)
                self._joint_qvel_indices.append(qvel_adr)
        
        self._num_actions = self._mj_model.nu
        self._obs_dim = self._compute_obs_dim()

    def _compute_obs_dim(self) -> int:
        num_joints = len(self._joint_qpos_indices)
        return (
            num_joints +  # joint positions
            num_joints +  # joint velocities
            4 +           # base quaternion
            3 +           # base angular velocity
            3 +           # command velocity
            3             # projected gravity
        )

    @property
    def observation_size(self) -> int:
        return self._obs_dim

    @property
    def action_size(self) -> int:
        return self._num_actions

    @property
    def backend(self) -> str:
        return "mjx"

    def reset(self, rng: jax.Array) -> BraxState:
        rng, cmd_key = jax.random.split(rng)
        
        data = mjx.make_data(self._mjx_model)
        qpos = reset_qpos(self._default_qpos, self._env_config.initial_height)
        data = data.replace(qpos=qpos, qvel=jnp.zeros_like(data.qvel))
        data = mjx.forward(self._mjx_model, data)
        
        command = self._sample_command(cmd_key)
        
        obs = self._compute_obs(data, command)
        
        metrics = {
            "episode_reward": jnp.array(0.0),
            "episode_length": jnp.array(0.0),
        }
        
        info = {
            "command": command,
            "prev_action": jnp.zeros(self._num_actions),
            "step_count": jnp.array(0, dtype=jnp.int32),
            "rng": rng,
            "truncation": jnp.array(0.0),
        }
        
        return BraxState(
            pipeline_state=data,
            obs=obs,
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
            metrics=metrics,
            info=info,
        )

    def step(self, state: BraxState, action: jnp.ndarray) -> BraxState:
        rng = state.info["rng"]
        rng, cmd_key = jax.random.split(rng, 2)
        
        scaled_action = action * self._env_config.action_scale
        scaled_action = jnp.clip(self._default_ctrl + scaled_action, self._ctrl_min, self._ctrl_max)
        
        def physics_step(data, _):
            data = data.replace(ctrl=scaled_action)
            data = mjx.step(self._mjx_model, data)
            return data, None
        
        mjx_data, _ = jax.lax.scan(
            physics_step,
            state.pipeline_state,
            None,
            length=self._env_config.control_decimation,
        )
        
        obs = self._compute_obs(mjx_data, state.info["command"])
        
        done = self._compute_termination(mjx_data)

        reward_components = self._compute_reward_components(
            mjx_data=mjx_data,
            action=action,
            prev_action=state.info["prev_action"],
            command=state.info["command"],
            done=done,
        )
        reward = reward_components["total_reward"]
        
        step_count = state.info["step_count"] + 1
        truncated = step_count >= self._env_config.episode_length
        done = done | truncated
        
        resample_interval = int(
            self._env_config.command_resample_time / 
            (self._env_config.dt * self._env_config.control_decimation)
        )
        should_resample = (step_count % resample_interval) == 0
        new_command = jnp.where(
            should_resample,
            self._sample_command(cmd_key),
            state.info["command"],
        )
        
        metrics = {
            **state.metrics,
            "reward/velocity_tracking": reward_components["velocity_tracking"],
            "reward/yaw_tracking": reward_components["yaw_tracking"],
            "reward/upright": reward_components["upright"],
            "reward/height": reward_components["height"],
            "reward/energy": reward_components["energy"],
            "reward/smoothness": reward_components["smoothness"],
        }
        
        info = {
            **state.info,
            "command": new_command,
            "prev_action": action,
            "step_count": step_count,
            "rng": rng,
            "truncation": jnp.where(truncated, 1.0, 0.0),
        }
        
        return BraxState(
            pipeline_state=mjx_data,
            obs=obs,
            reward=reward,
            done=jnp.where(done, 1.0, 0.0),
            metrics=metrics,
            info=info,
        )

    def _compute_obs(self, mjx_data: mjx.Data, command: jnp.ndarray) -> jnp.ndarray:
        return compute_observation(
            mjx_data.qpos,
            mjx_data.qvel,
            jnp.array(self._joint_qpos_indices),
            jnp.array(self._joint_qvel_indices),
            command,
        )

    def _sample_command(self, rng: jax.Array) -> jnp.ndarray:
        return sample_command_single(rng, self._env_config)

    def _compute_termination(self, mjx_data: mjx.Data) -> jnp.ndarray:
        done, _ = compute_termination(mjx_data.qpos, self._termination_config)
        return done

    def _compute_reward_components(
        self,
        mjx_data: mjx.Data,
        action: jnp.ndarray,
        prev_action: jnp.ndarray,
        command: jnp.ndarray,
        done: jnp.ndarray | None = None,
    ) -> dict[str, jnp.ndarray]:
        components = compute_reward_components(
            mjx_data=jax.tree.map(lambda x: x[None, ...], mjx_data),
            action=action[None, ...],
            prev_action=prev_action[None, ...],
            command=command[None, ...],
            joint_qpos_indices=jnp.array(self._joint_qpos_indices),
            joint_qvel_indices=jnp.array(self._joint_qvel_indices),
            done=None if done is None else done[None, ...],
            config=self._reward_config,
        )
        return {key: value[0] for key, value in components.items()}

    def set_command(self, state: BraxState, command: jnp.ndarray) -> BraxState:
        obs = self._compute_obs(state.pipeline_state, command)
        info = {**state.info, "command": command}
        return state.replace(obs=obs, info=info)

    def diagnostics_summary(self) -> dict[str, Any]:
        return {
            "actuators": actuator_mode_summary(self._mj_model),
            "default_ctrl": jnp.array(self._default_ctrl).tolist(),
            "initial_height": self._env_config.initial_height,
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
