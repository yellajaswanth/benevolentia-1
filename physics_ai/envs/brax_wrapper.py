from __future__ import annotations

import functools
from typing import Any, Dict

import flax.struct
import jax
import jax.numpy as jnp
from brax.envs import base as brax_base
import mujoco
from mujoco import mjx

from physics_ai.envs.h1_env import EnvConfig
from physics_ai.envs.domain_rand import DomainRandomizationConfig
from physics_ai.rewards.walking import RewardConfig
from physics_ai.utils.jax_utils import quat_rotate_inverse


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
    ):
        self._env_config = env_config or EnvConfig()
        self._reward_config = reward_config or RewardConfig()
        self._dr_config = dr_config
        
        if asset_path is None:
            from pathlib import Path
            asset_path = Path(__file__).parent.parent.parent / "assets" / "unitree_h1" / "h1.xml"
        
        self._mj_model = mujoco.MjModel.from_xml_path(str(asset_path))
        self._mj_model.opt.timestep = self._env_config.dt
        self._mjx_model = mjx.put_model(self._mj_model)
        
        self._setup_indices()
        self._default_qpos = jnp.array(self._mj_model.qpos0)

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
        qpos = self._default_qpos.at[2].set(1.0)
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
            "truncation": jnp.array(False),
        }
        
        return BraxState(
            pipeline_state=data,
            obs=obs,
            reward=jnp.array(0.0),
            done=jnp.array(False),
            metrics=metrics,
            info=info,
        )

    def step(self, state: BraxState, action: jnp.ndarray) -> BraxState:
        rng = state.info["rng"]
        rng, cmd_key, reset_key = jax.random.split(rng, 3)
        
        scaled_action = action * self._env_config.action_scale
        
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
        
        reward = self._compute_reward(
            mjx_data=mjx_data,
            action=action,
            prev_action=state.info["prev_action"],
            command=state.info["command"],
        )
        
        done = self._compute_termination(mjx_data)
        
        step_count = state.info["step_count"] + 1
        truncated = step_count >= self._env_config.episode_length
        done = done | truncated
        
        resample_interval = int(
            self._env_config.command_resample_time / 
            (self._env_config.dt * self._env_config.control_decimation)
        )
        should_resample = (step_count % resample_interval) == 0
        new_command = jax.lax.cond(
            should_resample,
            lambda: self._sample_command(cmd_key),
            lambda: state.info["command"],
        )
        
        reset_state = self.reset(reset_key)
        
        mjx_data = jax.lax.cond(
            done,
            lambda: reset_state.pipeline_state,
            lambda: mjx_data,
        )
        obs = jax.lax.cond(
            done,
            lambda: reset_state.obs,
            lambda: obs,
        )
        new_command = jax.lax.cond(
            done,
            lambda: reset_state.info["command"],
            lambda: new_command,
        )
        step_count = jax.lax.cond(
            done,
            lambda: jnp.array(0, dtype=jnp.int32),
            lambda: step_count,
        )
        
        episode_reward = state.metrics["episode_reward"] + reward
        episode_length = state.metrics["episode_length"] + 1
        
        episode_reward = jax.lax.cond(done, lambda: jnp.array(0.0), lambda: episode_reward)
        episode_length = jax.lax.cond(done, lambda: jnp.array(0.0), lambda: episode_length)
        
        metrics = {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
        }
        
        info = {
            "command": new_command,
            "prev_action": action,
            "step_count": step_count,
            "rng": rng,
            "truncation": truncated,
        }
        
        return BraxState(
            pipeline_state=mjx_data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def _compute_obs(self, mjx_data: mjx.Data, command: jnp.ndarray) -> jnp.ndarray:
        qpos = mjx_data.qpos
        qvel = mjx_data.qvel
        
        joint_pos = qpos[jnp.array(self._joint_qpos_indices)]
        joint_vel = qvel[jnp.array(self._joint_qvel_indices)]
        
        base_quat = qpos[3:7]
        base_ang_vel = qvel[3:6]
        
        gravity_world = jnp.array([0.0, 0.0, -1.0])
        projected_gravity = quat_rotate_inverse(base_quat, gravity_world)
        
        obs = jnp.concatenate([
            joint_pos,
            joint_vel,
            base_quat,
            base_ang_vel,
            command,
            projected_gravity,
        ])
        
        return obs

    def _sample_command(self, rng: jax.Array) -> jnp.ndarray:
        rng, *keys = jax.random.split(rng, 4)
        
        vx = jax.random.uniform(
            keys[0], (),
            minval=self._env_config.vx_range[0],
            maxval=self._env_config.vx_range[1],
        )
        vy = jax.random.uniform(
            keys[1], (),
            minval=self._env_config.vy_range[0],
            maxval=self._env_config.vy_range[1],
        )
        vyaw = jax.random.uniform(
            keys[2], (),
            minval=self._env_config.vyaw_range[0],
            maxval=self._env_config.vyaw_range[1],
        )
        
        return jnp.stack([vx, vy, vyaw])

    def _compute_termination(self, mjx_data: mjx.Data) -> jnp.ndarray:
        qpos = mjx_data.qpos
        
        base_height = qpos[2]
        base_quat = qpos[3:7]
        
        gravity_world = jnp.array([0.0, 0.0, -1.0])
        projected_gravity = quat_rotate_inverse(base_quat, gravity_world)
        
        pitch = jnp.arcsin(-projected_gravity[0])
        roll = jnp.arctan2(projected_gravity[1], projected_gravity[2])
        
        fallen = base_height < 0.3
        tilted = (jnp.abs(pitch) > 0.5) | (jnp.abs(roll) > 0.5)
        
        return fallen | tilted

    def _compute_reward(
        self,
        mjx_data: mjx.Data,
        action: jnp.ndarray,
        prev_action: jnp.ndarray,
        command: jnp.ndarray,
    ) -> jnp.ndarray:
        qpos = mjx_data.qpos
        qvel = mjx_data.qvel
        
        base_pos = qpos[:3]
        base_quat = qpos[3:7]
        base_lin_vel = qvel[:3]
        base_ang_vel = qvel[3:6]
        
        base_lin_vel_local = quat_rotate_inverse(base_quat, base_lin_vel)
        base_ang_vel_local = quat_rotate_inverse(base_quat, base_ang_vel)
        
        vel_error = jnp.sum((base_lin_vel_local[:2] - command[:2]) ** 2)
        vel_tracking = jnp.exp(-vel_error / self._reward_config.velocity_tracking_scale)
        
        yaw_error = (base_ang_vel_local[2] - command[2]) ** 2
        yaw_tracking = jnp.exp(-yaw_error / self._reward_config.yaw_rate_scale)
        
        gravity_world = jnp.array([0.0, 0.0, -1.0])
        projected_gravity = quat_rotate_inverse(base_quat, gravity_world)
        upright = projected_gravity[2]
        
        height_error = jnp.abs(base_pos[2] - self._reward_config.target_height)
        height = jnp.exp(-height_error * 10.0)
        
        energy = jnp.sum(action ** 2)
        
        smoothness = jnp.sum((action - prev_action) ** 2)
        
        alive = self._reward_config.alive_bonus
        
        total_reward = (
            self._reward_config.velocity_tracking_weight * vel_tracking +
            self._reward_config.yaw_rate_weight * yaw_tracking +
            self._reward_config.upright_weight * upright +
            self._reward_config.height_weight * height +
            self._reward_config.energy_weight * energy +
            self._reward_config.smoothness_weight * smoothness +
            alive
        )
        
        return total_reward * self._reward_config.reward_scaling


def create_brax_h1_env(
    env_config: EnvConfig | None = None,
    reward_config: RewardConfig | None = None,
    dr_config: DomainRandomizationConfig | None = None,
) -> BraxH1EnvWrapper:
    return BraxH1EnvWrapper(
        env_config=env_config,
        reward_config=reward_config,
        dr_config=dr_config,
    )
