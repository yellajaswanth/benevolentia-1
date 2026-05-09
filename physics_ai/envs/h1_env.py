from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from physics_ai.envs.h1_shared import (
    build_default_ctrl,
    build_default_qpos,
    compute_observation,
    compute_termination,
    default_asset_path,
    reset_qpos,
    sample_commands_batched,
)
from physics_ai.envs.h1_structs import EnvConfig
from physics_ai.rewards.walking import RewardConfig
from physics_ai.config import TerminationConfig


class EnvState(NamedTuple):
    mjx_data: mjx.Data
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    info: dict[str, Any]
    command: jnp.ndarray
    prev_action: jnp.ndarray
    step_count: jnp.ndarray
    rng: jax.Array


class UnitreeH1Env:
    def __init__(
        self,
        config: EnvConfig | None = None,
        asset_path: str | Path | None = None,
        reward_config: RewardConfig | None = None,
        termination_config: TerminationConfig | None = None,
    ):
        self.config = config or EnvConfig()
        self.reward_config = reward_config or RewardConfig()
        self.termination_config = termination_config or TerminationConfig()
        
        if asset_path is None:
            asset_path = default_asset_path()
        self.asset_path = Path(asset_path)
        
        self.mj_model = mujoco.MjModel.from_xml_path(str(self.asset_path))
        self.mj_model.opt.timestep = self.config.dt
        
        self.mjx_model = mjx.put_model(self.mj_model)
        
        self._setup_indices()
        
        self.num_actions = len(self.actuator_indices)
        self.obs_dim = self._compute_obs_dim()
        
        self._default_qpos = self._get_default_qpos()
        self._default_ctrl = build_default_ctrl(self.mj_model, self.config.default_joint_angles)
        self._ctrl_min = jnp.array(self.mj_model.actuator_ctrlrange[:, 0])
        self._ctrl_max = jnp.array(self.mj_model.actuator_ctrlrange[:, 1])

    def _setup_indices(self) -> None:
        self.actuator_indices = list(range(self.mj_model.nu))
        
        self.base_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        
        self.joint_qpos_indices = []
        self.joint_qvel_indices = []
        
        for i in range(self.mj_model.njnt):
            jnt_type = self.mj_model.jnt_type[i]
            if jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
                qpos_adr = self.mj_model.jnt_qposadr[i]
                qvel_adr = self.mj_model.jnt_dofadr[i]
                self.joint_qpos_indices.append(qpos_adr)
                self.joint_qvel_indices.append(qvel_adr)

    def _compute_obs_dim(self) -> int:
        num_joints = len(self.joint_qpos_indices)
        return (
            num_joints +  # joint positions
            num_joints +  # joint velocities
            4 +           # base quaternion
            3 +           # base angular velocity
            3 +           # command velocity
            3             # projected gravity
        )

    def _get_default_qpos(self) -> jnp.ndarray:
        return build_default_qpos(self.mj_model, self.config.default_joint_angles)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array) -> EnvState:
        rng, *keys = jax.random.split(rng, 4)
        
        batch_keys = jax.random.split(keys[0], self.config.num_envs)
        
        def reset_single(key):
            data = mjx.make_data(self.mjx_model)
            
            qpos = reset_qpos(self._default_qpos, self.config.initial_height)
            
            data = data.replace(qpos=qpos, qvel=jnp.zeros_like(data.qvel))
            data = mjx.forward(self.mjx_model, data)
            return data
        
        mjx_data = jax.vmap(reset_single)(batch_keys)
        
        command = self._sample_commands(keys[1], self.config.num_envs)
        
        prev_action = jnp.zeros((self.config.num_envs, self.num_actions))
        step_count = jnp.zeros(self.config.num_envs, dtype=jnp.int32)
        
        obs = self._compute_obs(mjx_data, command)
        
        return EnvState(
            mjx_data=mjx_data,
            obs=obs,
            reward=jnp.zeros(self.config.num_envs),
            done=jnp.zeros(self.config.num_envs, dtype=jnp.bool_),
            info={"truncated": jnp.zeros(self.config.num_envs, dtype=jnp.bool_)},
            command=command,
            prev_action=prev_action,
            step_count=step_count,
            rng=keys[2],
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, action: jnp.ndarray) -> EnvState:
        rng, step_key, cmd_key, reset_key = jax.random.split(state.rng, 4)
        
        scaled_action = action * self.config.action_scale
        ctrl = jnp.clip(self._default_ctrl + scaled_action, self._ctrl_min, self._ctrl_max)
        
        def single_env_step(single_data, single_ctrl):
            def physics_step(data, _):
                data = data.replace(ctrl=single_ctrl)
                data = mjx.step(self.mjx_model, data)
                return data, None
            
            result, _ = jax.lax.scan(
                physics_step,
                single_data,
                None,
                length=self.config.control_decimation,
            )
            return result
        
        mjx_data = jax.vmap(single_env_step)(state.mjx_data, ctrl)
        
        done = self._compute_termination(mjx_data)
        obs = self._compute_obs(mjx_data, state.command)
        
        from physics_ai.rewards.walking import compute_reward
        reward = compute_reward(
            mjx_data=mjx_data,
            action=action,
            prev_action=state.prev_action,
            command=state.command,
            joint_qpos_indices=jnp.array(self.joint_qpos_indices),
            joint_qvel_indices=jnp.array(self.joint_qvel_indices),
            done=done,
            config=self.reward_config,
        )
        
        step_count = state.step_count + 1
        truncated = step_count >= self.config.episode_length
        done = done | truncated
        
        resample_interval = int(self.config.command_resample_time / (self.config.dt * self.config.control_decimation))
        should_resample = (step_count % resample_interval) == 0
        new_command = jax.lax.cond(
            jnp.any(should_resample),
            lambda: jnp.where(
                should_resample[:, None],
                self._sample_commands(cmd_key, self.config.num_envs),
                state.command,
            ),
            lambda: state.command,
        )
        
        reset_keys = jax.random.split(reset_key, self.config.num_envs)
        mjx_data, obs, new_command, step_count = self._reset_done_envs(
            mjx_data, obs, new_command, step_count, done, reset_keys
        )
        
        return EnvState(
            mjx_data=mjx_data,
            obs=obs,
            reward=reward,
            done=done,
            info={"truncated": truncated},
            command=new_command,
            prev_action=action,
            step_count=jnp.where(done, jnp.zeros_like(step_count), step_count),
            rng=rng,
        )

    def _reset_done_envs(
        self,
        mjx_data: mjx.Data,
        obs: jnp.ndarray,
        command: jnp.ndarray,
        step_count: jnp.ndarray,
        done: jnp.ndarray,
        reset_keys: jax.Array,
    ):
        def reset_single(key):
            data = mjx.make_data(self.mjx_model)
            qpos = reset_qpos(self._default_qpos, self.config.initial_height)
            data = data.replace(qpos=qpos, qvel=jnp.zeros_like(data.qvel))
            data = mjx.forward(self.mjx_model, data)
            return data
        
        reset_data = jax.vmap(reset_single)(reset_keys)
        
        def select(reset_val, current_val, done_mask):
            return jnp.where(done_mask.reshape(-1, *([1] * (reset_val.ndim - 1))), reset_val, current_val)
        
        mjx_data = jax.tree.map(lambda r, c: select(r, c, done), reset_data, mjx_data)
        
        reset_commands = self._sample_commands(reset_keys[0], self.config.num_envs)
        command = jnp.where(done[:, None], reset_commands, command)
        
        obs = self._compute_obs(mjx_data, command)
        step_count = jnp.where(done, jnp.zeros_like(step_count), step_count)
        
        return mjx_data, obs, command, step_count

    def _compute_obs(self, mjx_data: mjx.Data, command: jnp.ndarray) -> jnp.ndarray:
        return compute_observation(
            mjx_data.qpos,
            mjx_data.qvel,
            jnp.array(self.joint_qpos_indices),
            jnp.array(self.joint_qvel_indices),
            command,
        )

    def _sample_commands(self, rng: jax.Array, num_envs: int) -> jnp.ndarray:
        return sample_commands_batched(rng, num_envs, self.config)

    def _compute_termination(self, mjx_data: mjx.Data) -> jnp.ndarray:
        done, _ = compute_termination(mjx_data.qpos, self.termination_config)
        return done

    @property
    def observation_space_shape(self) -> tuple[int, ...]:
        return (self.obs_dim,)

    @property
    def action_space_shape(self) -> tuple[int, ...]:
        return (self.num_actions,)
