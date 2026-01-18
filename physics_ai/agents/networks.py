from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class ActorCritic(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    log_std_min: float = -2.0
    log_std_max: float = 2.0
    
    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        x = obs
        for dim in self.hidden_dims:
            x = nn.Dense(dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
            x = nn.elu(x)
        
        mean = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
        )(x)
        
        log_std = self.param(
            "log_std",
            nn.initializers.constant(-1.0),
            (self.action_dim,),
        )
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def get_action(
        self,
        params,
        obs: jnp.ndarray,
        rng: jax.Array,
        deterministic: bool = False,
    ):
        mean, log_std = self.apply(params, obs)
        std = jnp.exp(log_std)
        
        if deterministic:
            return mean, None
        
        noise = jax.random.normal(rng, mean.shape)
        action = mean + noise * std
        
        log_prob = -0.5 * (
            jnp.sum(((action - mean) / std) ** 2, axis=-1) +
            jnp.sum(jnp.log(std), axis=-1) * 2 +
            self.action_dim * jnp.log(2 * jnp.pi)
        )
        
        return action, log_prob


class Critic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    
    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = obs
        for dim in self.hidden_dims:
            x = nn.Dense(dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
            x = nn.elu(x)
        
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)
        return value.squeeze(-1)


class ActorCriticSeparate(nn.Module):
    action_dim: int
    actor_hidden_dims: Sequence[int] = (256, 256)
    critic_hidden_dims: Sequence[int] = (256, 256)
    log_std_min: float = -2.0
    log_std_max: float = 2.0
    
    def setup(self):
        self.actor = Actor(
            action_dim=self.action_dim,
            hidden_dims=self.actor_hidden_dims,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
        )
        self.critic = Critic(hidden_dims=self.critic_hidden_dims)
    
    def __call__(self, obs: jnp.ndarray):
        mean, log_std = self.actor(obs)
        value = self.critic(obs)
        return mean, log_std, value


class Actor(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    log_std_min: float = -2.0
    log_std_max: float = 2.0
    
    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        x = obs
        for dim in self.hidden_dims:
            x = nn.Dense(dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
            x = nn.elu(x)
        
        mean = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
        )(x)
        
        log_std = self.param(
            "log_std",
            nn.initializers.constant(-1.0),
            (self.action_dim,),
        )
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

