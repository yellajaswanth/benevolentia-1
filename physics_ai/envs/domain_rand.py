from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx


@dataclass
class DomainRandomizationConfig:
    friction_enabled: bool = True
    friction_range: tuple[float, float] = (0.2, 1.0)
    
    mass_enabled: bool = True
    mass_scale_range: tuple[float, float] = (0.9, 1.1)
    
    motor_strength_enabled: bool = True
    motor_strength_range: tuple[float, float] = (0.85, 1.15)
    
    push_enabled: bool = True
    push_magnitude_range: tuple[float, float] = (0.0, 50.0)
    push_interval_range: tuple[float, float] = (5.0, 15.0)
    
    latency_enabled: bool = True
    latency_range_ms: tuple[int, int] = (0, 20)
    
    com_displacement_enabled: bool = False
    com_displacement_range: tuple[float, float] = (-0.05, 0.05)
    
    gravity_noise_enabled: bool = False
    gravity_noise_range: tuple[float, float] = (-0.1, 0.1)


@dataclass
class RandomizedParams:
    friction_scale: jnp.ndarray | None = None
    mass_scale: jnp.ndarray | None = None
    motor_strength_scale: jnp.ndarray | None = None
    push_force: jnp.ndarray | None = None
    push_interval: jnp.ndarray | None = None
    latency_steps: jnp.ndarray | None = None
    com_displacement: jnp.ndarray | None = None
    gravity_noise: jnp.ndarray | None = None


class DomainRandomizer:
    def __init__(
        self,
        config: DomainRandomizationConfig | None = None,
        num_envs: int = 4096,
    ):
        self.config = config or DomainRandomizationConfig()
        self.num_envs = num_envs
    
    def sample(self, rng: jax.Array) -> RandomizedParams:
        keys = jax.random.split(rng, 8)
        
        friction_scale = None
        if self.config.friction_enabled:
            friction_scale = jax.random.uniform(
                keys[0],
                (self.num_envs,),
                minval=self.config.friction_range[0],
                maxval=self.config.friction_range[1],
            )
        
        mass_scale = None
        if self.config.mass_enabled:
            mass_scale = jax.random.uniform(
                keys[1],
                (self.num_envs,),
                minval=self.config.mass_scale_range[0],
                maxval=self.config.mass_scale_range[1],
            )
        
        motor_strength_scale = None
        if self.config.motor_strength_enabled:
            motor_strength_scale = jax.random.uniform(
                keys[2],
                (self.num_envs,),
                minval=self.config.motor_strength_range[0],
                maxval=self.config.motor_strength_range[1],
            )
        
        push_force = None
        push_interval = None
        if self.config.push_enabled:
            push_force = jax.random.uniform(
                keys[3],
                (self.num_envs, 3),
                minval=-self.config.push_magnitude_range[1],
                maxval=self.config.push_magnitude_range[1],
            )
            push_interval = jax.random.uniform(
                keys[4],
                (self.num_envs,),
                minval=self.config.push_interval_range[0],
                maxval=self.config.push_interval_range[1],
            )
        
        latency_steps = None
        if self.config.latency_enabled:
            latency_steps = jax.random.randint(
                keys[5],
                (self.num_envs,),
                minval=self.config.latency_range_ms[0],
                maxval=self.config.latency_range_ms[1] + 1,
            )
        
        com_displacement = None
        if self.config.com_displacement_enabled:
            com_displacement = jax.random.uniform(
                keys[6],
                (self.num_envs, 3),
                minval=self.config.com_displacement_range[0],
                maxval=self.config.com_displacement_range[1],
            )
        
        gravity_noise = None
        if self.config.gravity_noise_enabled:
            gravity_noise = jax.random.uniform(
                keys[7],
                (self.num_envs, 3),
                minval=self.config.gravity_noise_range[0],
                maxval=self.config.gravity_noise_range[1],
            )
        
        return RandomizedParams(
            friction_scale=friction_scale,
            mass_scale=mass_scale,
            motor_strength_scale=motor_strength_scale,
            push_force=push_force,
            push_interval=push_interval,
            latency_steps=latency_steps,
            com_displacement=com_displacement,
            gravity_noise=gravity_noise,
        )
    
    @staticmethod
    def apply_friction(
        mjx_model: mjx.Model,
        friction_scale: jnp.ndarray,
        base_friction: jnp.ndarray,
    ) -> mjx.Model:
        scaled_friction = base_friction * friction_scale[:, None, None]
        return mjx_model.replace(geom_friction=scaled_friction)
    
    @staticmethod
    def apply_mass(
        mjx_model: mjx.Model,
        mass_scale: jnp.ndarray,
        base_mass: jnp.ndarray,
    ) -> mjx.Model:
        scaled_mass = base_mass * mass_scale[:, None]
        return mjx_model.replace(body_mass=scaled_mass)
    
    @staticmethod
    def apply_motor_strength(
        action: jnp.ndarray,
        motor_strength_scale: jnp.ndarray,
    ) -> jnp.ndarray:
        return action * motor_strength_scale[:, None]
    
    @staticmethod
    def apply_push_force(
        mjx_data: mjx.Data,
        push_force: jnp.ndarray,
        torso_body_id: int,
    ) -> mjx.Data:
        xfrc = mjx_data.xfrc_applied
        xfrc = xfrc.at[:, torso_body_id, :3].set(push_force)
        return mjx_data.replace(xfrc_applied=xfrc)
    
    @staticmethod
    def clear_push_force(
        mjx_data: mjx.Data,
        torso_body_id: int,
    ) -> mjx.Data:
        xfrc = mjx_data.xfrc_applied
        xfrc = xfrc.at[:, torso_body_id, :3].set(0.0)
        return mjx_data.replace(xfrc_applied=xfrc)


class ActionLatencyBuffer:
    def __init__(self, max_latency_steps: int, num_envs: int, action_dim: int):
        self.max_latency_steps = max_latency_steps
        self.num_envs = num_envs
        self.action_dim = action_dim
    
    def init_buffer(self) -> jnp.ndarray:
        return jnp.zeros((self.num_envs, self.max_latency_steps, self.action_dim))
    
    @staticmethod
    def push(
        buffer: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.concatenate([action[:, None, :], buffer[:, :-1, :]], axis=1)
    
    @staticmethod
    def get_delayed_action(
        buffer: jnp.ndarray,
        latency_steps: jnp.ndarray,
    ) -> jnp.ndarray:
        batch_indices = jnp.arange(buffer.shape[0])
        return buffer[batch_indices, latency_steps]

