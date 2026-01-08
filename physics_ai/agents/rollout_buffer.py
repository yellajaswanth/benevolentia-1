from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray
    next_obs: jnp.ndarray
    next_value: jnp.ndarray


class RolloutBatch(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    advantage: jnp.ndarray
    return_: jnp.ndarray
    old_log_prob: jnp.ndarray
    old_value: jnp.ndarray


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    next_values: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    T, B = rewards.shape
    
    advantages = jnp.zeros((T, B))
    last_gae = jnp.zeros(B)
    
    def scan_fn(last_gae, t):
        idx = T - 1 - t
        delta = rewards[idx] + gamma * next_values[idx] * (1 - dones[idx]) - values[idx]
        advantage = delta + gamma * gae_lambda * (1 - dones[idx]) * last_gae
        return advantage, advantage
    
    _, advantages_reversed = jax.lax.scan(scan_fn, last_gae, jnp.arange(T))
    advantages = jnp.flip(advantages_reversed, axis=0)
    
    returns = advantages + values
    
    return advantages, returns


def create_minibatches(
    batch: RolloutBatch,
    rng: jax.Array,
    num_minibatches: int,
) -> RolloutBatch:
    T, B = batch.obs.shape[:2]
    total_size = T * B
    
    flat_batch = jax.tree.map(lambda x: x.reshape(total_size, *x.shape[2:]), batch)
    
    permutation = jax.random.permutation(rng, total_size)
    shuffled = jax.tree.map(lambda x: x[permutation], flat_batch)
    
    minibatch_size = total_size // num_minibatches
    
    minibatches = jax.tree.map(
        lambda x: x[:num_minibatches * minibatch_size].reshape(num_minibatches, minibatch_size, *x.shape[1:]),
        shuffled,
    )
    
    return minibatches

