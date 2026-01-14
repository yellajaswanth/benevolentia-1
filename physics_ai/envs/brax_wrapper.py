from __future__ import annotations

import functools
from typing import Any, Dict, Tuple

import flax.struct
import jax
import jax.numpy as jnp
from brax.envs import base as brax_base
from mujoco import mjx

from physics_ai.envs.h1_env import UnitreeH1Env, EnvConfig, EnvState
from physics_ai.envs.domain_rand import (
    DomainRandomizer,
    DomainRandomizationConfig,
    RandomizedParams,
)
from physics_ai.rewards.walking import RewardConfig


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
    ):
        self._env_config = env_config or EnvConfig()
        self._reward_config = reward_config or RewardConfig()
        self._dr_config = dr_config
        
        self._h1_env = UnitreeH1Env(
            config=self._env_config,
            reward_config=self._reward_config,
        )
        
        self._domain_randomizer = None
        if dr_config is not None:
            self._domain_randomizer = DomainRandomizer(
                config=dr_config,
                num_envs=self._env_config.num_envs,
            )

    @property
    def observation_size(self) -> int:
        return self._h1_env.obs_dim

    @property
    def action_size(self) -> int:
        return self._h1_env.num_actions

    @property
    def backend(self) -> str:
        return "mjx"

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array) -> BraxState:
        rng, env_key, dr_key = jax.random.split(rng, 3)
        
        env_state = self._h1_env.reset(env_key)
        
        dr_params = None
        if self._domain_randomizer is not None:
            dr_params = self._domain_randomizer.sample(dr_key)
        
        metrics = {
            "episode_reward": jnp.zeros(self._env_config.num_envs),
            "episode_length": jnp.zeros(self._env_config.num_envs),
        }
        
        info = {
            "command": env_state.command,
            "prev_action": env_state.prev_action,
            "step_count": env_state.step_count,
            "rng": env_state.rng,
            "dr_params": dr_params,
            "truncation": jnp.zeros(self._env_config.num_envs, dtype=jnp.bool_),
        }
        
        return BraxState(
            pipeline_state=env_state.mjx_data,
            obs=env_state.obs,
            reward=env_state.reward,
            done=env_state.done,
            metrics=metrics,
            info=info,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: BraxState, action: jnp.ndarray) -> BraxState:
        rng = state.info["rng"]
        rng, step_key, cmd_key, reset_key, dr_key = jax.random.split(rng, 5)
        
        dr_params = state.info.get("dr_params")
        scaled_action = action
        if dr_params is not None and dr_params.motor_strength_scale is not None:
            scaled_action = DomainRandomizer.apply_motor_strength(
                action, dr_params.motor_strength_scale
            )
        
        scaled_action = scaled_action * self._env_config.action_scale
        
        def single_env_step(single_data, single_ctrl):
            def physics_step(data, _):
                data = data.replace(ctrl=single_ctrl)
                data = mjx.step(self._h1_env.mjx_model, data)
                return data, None
            
            result, _ = jax.lax.scan(
                physics_step,
                single_data,
                None,
                length=self._env_config.control_decimation,
            )
            return result
        
        mjx_data = jax.vmap(single_env_step)(state.pipeline_state, scaled_action)
        
        obs = self._compute_obs(mjx_data, state.info["command"])
        
        from physics_ai.rewards.walking import compute_reward
        reward = compute_reward(
            mjx_data=mjx_data,
            action=action,
            prev_action=state.info["prev_action"],
            command=state.info["command"],
            joint_qpos_indices=jnp.array(self._h1_env.joint_qpos_indices),
            joint_qvel_indices=jnp.array(self._h1_env.joint_qvel_indices),
            config=self._reward_config,
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
            jnp.any(should_resample),
            lambda: jnp.where(
                should_resample[:, None],
                self._sample_commands(cmd_key),
                state.info["command"],
            ),
            lambda: state.info["command"],
        )
        
        reset_keys = jax.random.split(reset_key, self._env_config.num_envs)
        mjx_data, obs, new_command, step_count, new_dr_params = self._reset_done_envs(
            mjx_data, obs, new_command, step_count, done, reset_keys, dr_key
        )
        
        episode_reward = state.metrics["episode_reward"] + reward
        episode_length = state.metrics["episode_length"] + 1
        
        episode_reward = jnp.where(done, jnp.zeros_like(episode_reward), episode_reward)
        episode_length = jnp.where(done, jnp.zeros_like(episode_length), episode_length)
        
        metrics = {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
        }
        
        info = {
            "command": new_command,
            "prev_action": action,
            "step_count": jnp.where(done, jnp.zeros_like(step_count), step_count),
            "rng": rng,
            "dr_params": new_dr_params if new_dr_params is not None else dr_params,
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
        from physics_ai.utils.jax_utils import quat_rotate_inverse
        
        qpos = mjx_data.qpos
        qvel = mjx_data.qvel
        
        joint_pos = qpos[:, jnp.array(self._h1_env.joint_qpos_indices)]
        joint_vel = qvel[:, jnp.array(self._h1_env.joint_qvel_indices)]
        
        base_quat = qpos[:, 3:7]
        base_ang_vel = qvel[:, 3:6]
        
        gravity_world = jnp.array([0.0, 0.0, -1.0])
        projected_gravity = jax.vmap(quat_rotate_inverse)(
            base_quat, jnp.tile(gravity_world, (qpos.shape[0], 1))
        )
        
        obs = jnp.concatenate([
            joint_pos,
            joint_vel,
            base_quat,
            base_ang_vel,
            command,
            projected_gravity,
        ], axis=-1)
        
        return obs

    def _sample_commands(self, rng: jax.Array) -> jnp.ndarray:
        rng, *keys = jax.random.split(rng, 4)
        
        vx = jax.random.uniform(
            keys[0], (self._env_config.num_envs,),
            minval=self._env_config.vx_range[0],
            maxval=self._env_config.vx_range[1],
        )
        vy = jax.random.uniform(
            keys[1], (self._env_config.num_envs,),
            minval=self._env_config.vy_range[0],
            maxval=self._env_config.vy_range[1],
        )
        vyaw = jax.random.uniform(
            keys[2], (self._env_config.num_envs,),
            minval=self._env_config.vyaw_range[0],
            maxval=self._env_config.vyaw_range[1],
        )
        
        return jnp.stack([vx, vy, vyaw], axis=-1)

    def _compute_termination(self, mjx_data: mjx.Data) -> jnp.ndarray:
        from physics_ai.utils.jax_utils import quat_rotate_inverse
        
        qpos = mjx_data.qpos
        
        base_height = qpos[:, 2]
        base_quat = qpos[:, 3:7]
        
        gravity_world = jnp.array([0.0, 0.0, -1.0])
        projected_gravity = jax.vmap(quat_rotate_inverse)(
            base_quat, jnp.tile(gravity_world, (qpos.shape[0], 1))
        )
        
        pitch = jnp.arcsin(-projected_gravity[:, 0])
        roll = jnp.arctan2(projected_gravity[:, 1], projected_gravity[:, 2])
        
        fallen = base_height < 0.3
        tilted = (jnp.abs(pitch) > 0.5) | (jnp.abs(roll) > 0.5)
        
        return fallen | tilted

    def _reset_done_envs(
        self,
        mjx_data: mjx.Data,
        obs: jnp.ndarray,
        command: jnp.ndarray,
        step_count: jnp.ndarray,
        done: jnp.ndarray,
        reset_keys: jax.Array,
        dr_key: jax.Array,
    ) -> Tuple[mjx.Data, jnp.ndarray, jnp.ndarray, jnp.ndarray, RandomizedParams | None]:
        def reset_single(key):
            data = mjx.make_data(self._h1_env.mjx_model)
            qpos = self._h1_env._default_qpos.at[2].set(1.0)
            data = data.replace(qpos=qpos, qvel=jnp.zeros_like(data.qvel))
            data = mjx.forward(self._h1_env.mjx_model, data)
            return data
        
        reset_data = jax.vmap(reset_single)(reset_keys)
        
        def select(reset_val, current_val, done_mask):
            return jnp.where(
                done_mask.reshape(-1, *([1] * (reset_val.ndim - 1))),
                reset_val,
                current_val,
            )
        
        mjx_data = jax.tree.map(lambda r, c: select(r, c, done), reset_data, mjx_data)
        
        reset_commands = self._sample_commands(reset_keys[0])
        command = jnp.where(done[:, None], reset_commands, command)
        
        obs = self._compute_obs(mjx_data, command)
        step_count = jnp.where(done, jnp.zeros_like(step_count), step_count)
        
        new_dr_params = None
        if self._domain_randomizer is not None:
            new_dr_params = self._domain_randomizer.sample(dr_key)
            old_dr_params = None
        
        return mjx_data, obs, command, step_count, new_dr_params


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
