#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import yaml

from physics_ai.agents.ppo import PPOAgent, PPOConfig, PPOState
from physics_ai.agents.rollout_buffer import compute_gae, RolloutBatch
from physics_ai.envs.h1_env import UnitreeH1Env, EnvConfig, EnvState
from physics_ai.envs.domain_rand import DomainRandomizer, DomainRandomizationConfig


class TrainingConfig(NamedTuple):
    total_timesteps: int = 1_000_000_000
    rollout_length: int = 64
    eval_interval: int = 100
    save_interval: int = 500
    log_interval: int = 10
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


def load_config(config_path: str) -> tuple[EnvConfig, PPOConfig, DomainRandomizationConfig, TrainingConfig]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    env_config = EnvConfig(
        num_envs=cfg.get("env", {}).get("num_envs", 4096),
        episode_length=cfg.get("env", {}).get("episode_length", 1000),
        dt=cfg.get("env", {}).get("dt", 0.005),
        control_decimation=cfg.get("env", {}).get("control_decimation", 4),
        vx_range=tuple(cfg.get("commands", {}).get("vx_range", [-1.0, 1.0])),
        vy_range=tuple(cfg.get("commands", {}).get("vy_range", [-0.5, 0.5])),
        vyaw_range=tuple(cfg.get("commands", {}).get("vyaw_range", [-1.0, 1.0])),
        command_resample_time=cfg.get("commands", {}).get("resample_time", 10.0),
    )
    
    ppo_cfg = cfg.get("ppo", {})
    ppo_config = PPOConfig(
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        clip_ratio=ppo_cfg.get("clip_ratio", 0.2),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        entropy_coef=ppo_cfg.get("entropy_coef", 0.01),
        value_coef=ppo_cfg.get("value_coef", 0.5),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 1.0),
        num_epochs=ppo_cfg.get("num_epochs", 4),
        num_minibatches=ppo_cfg.get("num_minibatches", 4),
        normalize_advantages=ppo_cfg.get("normalize_advantages", True),
    )
    
    dr_cfg = cfg.get("domain_randomization", {})
    dr_config = DomainRandomizationConfig(
        friction_enabled=dr_cfg.get("friction", {}).get("enabled", True),
        friction_range=tuple(dr_cfg.get("friction", {}).get("range", [0.2, 1.0])),
        mass_enabled=dr_cfg.get("mass", {}).get("enabled", True),
        mass_scale_range=tuple(dr_cfg.get("mass", {}).get("scale_range", [0.9, 1.1])),
        motor_strength_enabled=dr_cfg.get("motor_strength", {}).get("enabled", True),
        motor_strength_range=tuple(dr_cfg.get("motor_strength", {}).get("scale_range", [0.85, 1.15])),
        push_enabled=dr_cfg.get("push_force", {}).get("enabled", True),
        push_magnitude_range=tuple(dr_cfg.get("push_force", {}).get("magnitude_range", [0.0, 50.0])),
        push_interval_range=tuple(dr_cfg.get("push_force", {}).get("interval", [5.0, 15.0])),
        latency_enabled=dr_cfg.get("latency", {}).get("enabled", True),
        latency_range_ms=tuple(dr_cfg.get("latency", {}).get("range_ms", [0, 20])),
    )
    
    train_cfg = cfg.get("training", {})
    train_config = TrainingConfig(
        total_timesteps=train_cfg.get("total_timesteps", 1_000_000_000),
        rollout_length=train_cfg.get("rollout_length", 64),
        eval_interval=train_cfg.get("eval_interval", 100),
        save_interval=train_cfg.get("save_interval", 500),
        log_interval=train_cfg.get("log_interval", 10),
        seed=train_cfg.get("seed", 42),
    )
    
    return env_config, ppo_config, dr_config, train_config


class Rollout(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray


def collect_rollout(
    env: UnitreeH1Env,
    agent: PPOAgent,
    state: PPOState,
    env_state: EnvState,
    rollout_length: int,
) -> tuple[Rollout, EnvState, PPOState]:
    def step_fn(carry, _):
        env_state, ppo_state = carry
        
        action, log_prob, value, rng = agent.get_action(ppo_state, env_state.obs)
        ppo_state = ppo_state._replace(rng=rng)
        
        next_env_state = env.step(env_state, action)
        
        transition = (
            env_state.obs,
            action,
            next_env_state.reward,
            next_env_state.done,
            value,
            log_prob,
        )
        
        return (next_env_state, ppo_state), transition
    
    (env_state, ppo_state), transitions = jax.lax.scan(
        step_fn,
        (env_state, state),
        None,
        length=rollout_length,
    )
    
    obs, action, reward, done, value, log_prob = transitions
    
    rollout = Rollout(
        obs=obs,
        action=action,
        reward=reward,
        done=done,
        value=value,
        log_prob=log_prob,
    )
    
    return rollout, env_state, ppo_state


def train(
    env_config: EnvConfig,
    ppo_config: PPOConfig,
    dr_config: DomainRandomizationConfig,
    train_config: TrainingConfig,
):
    rng = jax.random.PRNGKey(train_config.seed)
    rng, env_key, agent_key = jax.random.split(rng, 3)
    
    print("Initializing environment...")
    env = UnitreeH1Env(config=env_config)
    
    print("Initializing PPO agent...")
    agent = PPOAgent(
        obs_dim=env.obs_dim,
        action_dim=env.num_actions,
        config=ppo_config,
        rng=agent_key,
    )
    
    print("Initializing domain randomizer...")
    domain_randomizer = DomainRandomizer(config=dr_config, num_envs=env_config.num_envs)
    
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)
    
    print("Resetting environment...")
    env_state = env.reset(env_key)
    ppo_state = agent.state
    
    timesteps_per_update = train_config.rollout_length * env_config.num_envs
    num_updates = train_config.total_timesteps // timesteps_per_update
    
    print(f"\nTraining Configuration:")
    print(f"  Total timesteps: {train_config.total_timesteps:,}")
    print(f"  Num environments: {env_config.num_envs}")
    print(f"  Rollout length: {train_config.rollout_length}")
    print(f"  Timesteps per update: {timesteps_per_update:,}")
    print(f"  Total updates: {num_updates:,}")
    print()
    
    @functools.partial(jax.jit, static_argnums=(0, 1))
    def training_step(env, agent, ppo_state, env_state, rollout_length, gamma, gae_lambda):
        rollout, env_state, ppo_state = collect_rollout(
            env, agent, ppo_state, env_state, rollout_length
        )
        
        next_value = agent.get_value(ppo_state, env_state.obs)
        
        next_values = jnp.concatenate([rollout.value[1:], next_value[None]], axis=0)
        
        advantages, returns = compute_gae(
            rewards=rollout.reward,
            values=rollout.value,
            next_values=next_values,
            dones=rollout.done,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        
        batch = RolloutBatch(
            obs=rollout.obs,
            action=rollout.action,
            advantage=advantages,
            return_=returns,
            old_log_prob=rollout.log_prob,
            old_value=rollout.value,
        )
        
        ppo_state, metrics = agent._update_epochs(ppo_state, batch)
        
        mean_reward = jnp.mean(rollout.reward)
        mean_episode_length = jnp.mean(jnp.sum(~rollout.done, axis=0))
        
        metrics = {
            **metrics,
            "mean_reward": mean_reward,
            "mean_episode_length": mean_episode_length,
        }
        
        return ppo_state, env_state, metrics
    
    total_timesteps = 0
    start_time = time.time()
    
    print("Starting training...")
    
    for update in range(num_updates):
        update_start = time.time()
        
        ppo_state, env_state, metrics = training_step(
            env,
            agent,
            ppo_state,
            env_state,
            train_config.rollout_length,
            ppo_config.gamma,
            ppo_config.gae_lambda,
        )
        
        total_timesteps += timesteps_per_update
        
        if update % train_config.log_interval == 0:
            elapsed = time.time() - start_time
            fps = total_timesteps / elapsed
            update_time = time.time() - update_start
            
            print(
                f"Update {update:6d} | "
                f"Timesteps: {total_timesteps:12,} | "
                f"FPS: {fps:8.0f} | "
                f"Reward: {float(metrics['mean_reward']):7.3f} | "
                f"Actor Loss: {float(metrics['actor_loss']):8.4f} | "
                f"Critic Loss: {float(metrics['critic_loss']):8.4f} | "
                f"Entropy: {float(metrics['entropy']):7.3f} | "
                f"KL: {float(metrics['approx_kl']):7.4f}"
            )
        
        if update % train_config.save_interval == 0 and update > 0:
            checkpoint_path = os.path.join(
                train_config.checkpoint_dir,
                f"checkpoint_{update:06d}.pkl"
            )
            agent._state = ppo_state
            agent.save(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    final_path = os.path.join(train_config.checkpoint_dir, "final.pkl")
    agent._state = ppo_state
    agent.save(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")
    
    total_time = time.time() - start_time
    print(f"Total training time: {total_time / 3600:.2f} hours")
    print(f"Average FPS: {total_timesteps / total_time:.0f}")


def main():
    parser = argparse.ArgumentParser(description="Train H1 walking policy with PPO")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/h1_walking.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override number of parallel environments",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    args = parser.parse_args()
    
    config_path = Path(__file__).parent.parent / args.config
    env_config, ppo_config, dr_config, train_config = load_config(str(config_path))
    
    if args.seed is not None:
        train_config = train_config._replace(seed=args.seed)
    if args.num_envs is not None:
        env_config.num_envs = args.num_envs
    train_config = train_config._replace(checkpoint_dir=args.checkpoint_dir)
    
    train(env_config, ppo_config, dr_config, train_config)


if __name__ == "__main__":
    main()

