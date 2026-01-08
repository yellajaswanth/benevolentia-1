from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from physics_ai.agents.networks import ActorCritic, Critic
from physics_ai.agents.rollout_buffer import RolloutBatch, compute_gae, create_minibatches


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    num_epochs: int = 4
    num_minibatches: int = 4
    normalize_advantages: bool = True
    target_kl: float | None = None


class PPOState(NamedTuple):
    actor_state: TrainState
    critic_state: TrainState
    rng: jax.Array


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: PPOConfig | None = None,
        hidden_dims: tuple[int, ...] = (256, 256),
        rng: jax.Array | None = None,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config or PPOConfig()
        self.hidden_dims = hidden_dims
        
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        self.actor = ActorCritic(action_dim=action_dim, hidden_dims=hidden_dims)
        self.critic = Critic(hidden_dims=hidden_dims)
        
        self._state = self._init_state(rng)
    
    def _init_state(self, rng: jax.Array) -> PPOState:
        rng, actor_key, critic_key = jax.random.split(rng, 3)
        
        dummy_obs = jnp.zeros((1, self.obs_dim))
        
        actor_params = self.actor.init(actor_key, dummy_obs)
        critic_params = self.critic.init(critic_key, dummy_obs)
        
        actor_tx = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(self.config.learning_rate),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(self.config.learning_rate),
        )
        
        actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=actor_tx,
        )
        critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=critic_tx,
        )
        
        return PPOState(actor_state=actor_state, critic_state=critic_state, rng=rng)
    
    @property
    def state(self) -> PPOState:
        return self._state
    
    @state.setter
    def state(self, value: PPOState):
        self._state = value
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def get_action(
        self,
        state: PPOState,
        obs: jnp.ndarray,
        deterministic: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.Array]:
        rng, action_key = jax.random.split(state.rng)
        
        mean, log_std = self.actor.apply(state.actor_state.params, obs)
        value = self.critic.apply(state.critic_state.params, obs)
        
        std = jnp.exp(log_std)
        
        if deterministic:
            action = mean
            log_prob = jnp.zeros(obs.shape[0])
        else:
            noise = jax.random.normal(action_key, mean.shape)
            action = mean + noise * std
            
            log_prob = -0.5 * (
                jnp.sum(((action - mean) / std) ** 2, axis=-1) +
                jnp.sum(jnp.log(std)) * 2 +
                self.action_dim * jnp.log(2 * jnp.pi)
            )
        
        action = jnp.clip(action, -1.0, 1.0)
        
        return action, log_prob, value, rng
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def get_value(self, state: PPOState, obs: jnp.ndarray) -> jnp.ndarray:
        return self.critic.apply(state.critic_state.params, obs)
    
    def update(
        self,
        state: PPOState,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        dones: jnp.ndarray,
        old_log_probs: jnp.ndarray,
        old_values: jnp.ndarray,
        next_obs: jnp.ndarray,
    ) -> tuple[PPOState, dict[str, float]]:
        next_values = self.critic.apply(state.critic_state.params, next_obs[-1])
        
        all_values = jnp.concatenate([old_values, next_values[None]], axis=0)
        next_values_full = all_values[1:]
        
        advantages, returns = compute_gae(
            rewards=rewards,
            values=old_values,
            next_values=next_values_full,
            dones=dones,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        
        if self.config.normalize_advantages:
            advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        
        batch = RolloutBatch(
            obs=obs,
            action=actions,
            advantage=advantages,
            return_=returns,
            old_log_prob=old_log_probs,
            old_value=old_values,
        )
        
        state, metrics = self._update_epochs(state, batch)
        
        return state, metrics
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_epochs(
        self,
        state: PPOState,
        batch: RolloutBatch,
    ) -> tuple[PPOState, dict[str, jnp.ndarray]]:
        def epoch_step(carry, _):
            state, metrics_sum = carry
            rng, perm_key = jax.random.split(state.rng)
            
            minibatches = create_minibatches(batch, perm_key, self.config.num_minibatches)
            
            def minibatch_step(carry, minibatch):
                state = carry
                state, metrics = self._update_step(state, minibatch)
                return state, metrics
            
            state, epoch_metrics = jax.lax.scan(
                minibatch_step,
                state._replace(rng=rng),
                minibatches,
            )
            
            mean_metrics = jax.tree.map(jnp.mean, epoch_metrics)
            metrics_sum = jax.tree.map(lambda a, b: a + b, metrics_sum, mean_metrics)
            
            return (state, metrics_sum), None
        
        init_metrics = {
            "actor_loss": jnp.array(0.0),
            "critic_loss": jnp.array(0.0),
            "entropy": jnp.array(0.0),
            "approx_kl": jnp.array(0.0),
            "clip_fraction": jnp.array(0.0),
        }
        
        (state, metrics_sum), _ = jax.lax.scan(
            epoch_step,
            (state, init_metrics),
            None,
            length=self.config.num_epochs,
        )
        
        metrics = jax.tree.map(lambda x: x / self.config.num_epochs, metrics_sum)
        
        return state, metrics
    
    def _update_step(
        self,
        state: PPOState,
        minibatch: RolloutBatch,
    ) -> tuple[PPOState, dict[str, jnp.ndarray]]:
        def actor_loss_fn(actor_params):
            mean, log_std = self.actor.apply(actor_params, minibatch.obs)
            std = jnp.exp(log_std)
            
            log_prob = -0.5 * (
                jnp.sum(((minibatch.action - mean) / std) ** 2, axis=-1) +
                jnp.sum(jnp.log(std)) * 2 +
                self.action_dim * jnp.log(2 * jnp.pi)
            )
            
            ratio = jnp.exp(log_prob - minibatch.old_log_prob)
            
            clipped_ratio = jnp.clip(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
            
            surrogate1 = ratio * minibatch.advantage
            surrogate2 = clipped_ratio * minibatch.advantage
            policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))
            
            entropy = 0.5 * self.action_dim * (1 + jnp.log(2 * jnp.pi)) + jnp.sum(log_std)
            
            total_loss = policy_loss - self.config.entropy_coef * entropy
            
            approx_kl = jnp.mean((ratio - 1) - jnp.log(ratio))
            clip_fraction = jnp.mean(jnp.abs(ratio - 1) > self.config.clip_ratio)
            
            return total_loss, (policy_loss, entropy, approx_kl, clip_fraction)
        
        def critic_loss_fn(critic_params):
            values = self.critic.apply(critic_params, minibatch.obs)
            value_loss = jnp.mean((values - minibatch.return_) ** 2)
            return value_loss
        
        (actor_loss, (policy_loss, entropy, approx_kl, clip_fraction)), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(state.actor_state.params)
        
        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(state.critic_state.params)
        
        actor_state = state.actor_state.apply_gradients(grads=actor_grads)
        critic_state = state.critic_state.apply_gradients(grads=critic_grads)
        
        metrics = {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }
        
        return PPOState(actor_state=actor_state, critic_state=critic_state, rng=state.rng), metrics
    
    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "actor_params": self._state.actor_state.params,
                "critic_params": self._state.critic_state.params,
                "config": self.config,
            }, f)
    
    def load(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self._state = PPOState(
            actor_state=self._state.actor_state.replace(params=data["actor_params"]),
            critic_state=self._state.critic_state.replace(params=data["critic_params"]),
            rng=self._state.rng,
        )

