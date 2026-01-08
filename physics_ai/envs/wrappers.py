from __future__ import annotations

from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from physics_ai.envs.h1_env import UnitreeH1Env, EnvConfig, EnvState


class LocoMuJoCoWrapper(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    def __init__(
        self,
        env: UnitreeH1Env | None = None,
        config: EnvConfig | None = None,
        seed: int = 0,
        render_mode: str | None = None,
    ):
        super().__init__()
        
        if env is not None:
            self._env = env
        else:
            config = config or EnvConfig(num_envs=1)
            config.num_envs = 1  # Single env for gym compatibility
            self._env = UnitreeH1Env(config=config)
        
        self._seed = seed
        self._rng = jax.random.PRNGKey(seed)
        self._state: EnvState | None = None
        self.render_mode = render_mode
        
        obs_shape = self._env.observation_space_shape
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32,
        )
        
        action_shape = self._env.action_space_shape
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=action_shape,
            dtype=np.float32,
        )
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = jax.random.PRNGKey(seed)
        
        self._rng, reset_key = jax.random.split(self._rng)
        self._state = self._env.reset(reset_key)
        
        obs = np.array(self._state.obs[0])
        info = {
            "command": np.array(self._state.command[0]),
        }
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Environment must be reset before stepping")
        
        action_jax = jnp.array(action)[None, :]
        self._state = self._env.step(self._state, action_jax)
        
        obs = np.array(self._state.obs[0])
        reward = float(self._state.reward[0])
        terminated = bool(self._state.done[0])
        truncated = bool(self._state.info.get("truncated", jnp.array([False]))[0])
        
        info = {
            "command": np.array(self._state.command[0]),
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        
        if self._state is None:
            return None
        
        import mujoco
        
        renderer = mujoco.Renderer(self._env.mj_model, height=480, width=640)
        
        mj_data = mujoco.MjData(self._env.mj_model)
        mj_data.qpos[:] = np.array(self._state.mjx_data.qpos[0])
        mj_data.qvel[:] = np.array(self._state.mjx_data.qvel[0])
        mujoco.mj_forward(self._env.mj_model, mj_data)
        
        renderer.update_scene(mj_data, camera="track")
        frame = renderer.render()
        
        return frame
    
    def close(self):
        pass
    
    @property
    def unwrapped(self) -> UnitreeH1Env:
        return self._env


class VectorizedEnvWrapper:
    def __init__(
        self,
        env: UnitreeH1Env,
        seed: int = 0,
    ):
        self._env = env
        self._rng = jax.random.PRNGKey(seed)
        self._state: EnvState | None = None
    
    @property
    def num_envs(self) -> int:
        return self._env.config.num_envs
    
    @property
    def observation_space_shape(self) -> tuple[int, ...]:
        return self._env.observation_space_shape
    
    @property
    def action_space_shape(self) -> tuple[int, ...]:
        return self._env.action_space_shape
    
    def reset(self) -> jnp.ndarray:
        self._rng, reset_key = jax.random.split(self._rng)
        self._state = self._env.reset(reset_key)
        return self._state.obs
    
    def step(self, action: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        if self._state is None:
            raise RuntimeError("Environment must be reset before stepping")
        
        self._state = self._env.step(self._state, action)
        
        return (
            self._state.obs,
            self._state.reward,
            self._state.done,
            self._state.info,
        )
    
    @property
    def state(self) -> EnvState:
        if self._state is None:
            raise RuntimeError("Environment must be reset first")
        return self._state

