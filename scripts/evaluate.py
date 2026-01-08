#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from physics_ai.agents.ppo import PPOAgent, PPOConfig
from physics_ai.envs.h1_env import UnitreeH1Env, EnvConfig
from physics_ai.envs.wrappers import LocoMuJoCoWrapper


def evaluate_policy(
    agent: PPOAgent,
    env: UnitreeH1Env,
    num_episodes: int = 10,
    max_steps: int = 1000,
    render: bool = False,
    verbose: bool = True,
) -> dict:
    rng = jax.random.PRNGKey(0)
    
    episode_rewards = []
    episode_lengths = []
    velocities_achieved = []
    
    for ep in range(num_episodes):
        rng, reset_key = jax.random.split(rng)
        env_state = env.reset(reset_key)
        ppo_state = agent.state
        
        ep_reward = 0.0
        ep_length = 0
        ep_velocities = []
        
        for step in range(max_steps):
            action, _, _, new_rng = agent.get_action(
                ppo_state,
                env_state.obs,
                deterministic=True,
            )
            ppo_state = ppo_state._replace(rng=new_rng)
            
            env_state = env.step(env_state, action)
            
            ep_reward += float(env_state.reward[0])
            ep_length += 1
            
            base_vel = env_state.mjx_data.qvel[0, :2]
            ep_velocities.append(np.array(base_vel))
            
            if render:
                time.sleep(0.01)
            
            if env_state.done[0]:
                break
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        velocities_achieved.append(np.mean(ep_velocities, axis=0))
        
        if verbose:
            print(
                f"Episode {ep + 1:3d} | "
                f"Reward: {ep_reward:8.2f} | "
                f"Length: {ep_length:4d} | "
                f"Avg Vel: [{velocities_achieved[-1][0]:.2f}, {velocities_achieved[-1][1]:.2f}]"
            )
    
    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "mean_velocity": np.mean(velocities_achieved, axis=0),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }
    
    return results


def evaluate_with_gym(
    agent: PPOAgent,
    num_episodes: int = 10,
    max_steps: int = 1000,
    render: bool = False,
    seed: int = 0,
) -> dict:
    config = EnvConfig(num_envs=1)
    wrapper = LocoMuJoCoWrapper(config=config, seed=seed, render_mode="rgb_array" if render else None)
    
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs, info = wrapper.reset()
        
        ep_reward = 0.0
        ep_length = 0
        
        for step in range(max_steps):
            obs_jax = jnp.array(obs)[None, :]
            
            action, _, _, _ = agent.get_action(
                agent.state,
                obs_jax,
                deterministic=True,
            )
            action = np.array(action[0])
            
            obs, reward, terminated, truncated, info = wrapper.step(action)
            
            ep_reward += reward
            ep_length += 1
            
            if render:
                frame = wrapper.render()
            
            if terminated or truncated:
                break
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        
        print(
            f"Episode {ep + 1:3d} | "
            f"Reward: {ep_reward:8.2f} | "
            f"Length: {ep_length:4d}"
        )
    
    wrapper.close()
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
    }


def print_results(results: dict):
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Mean Reward:  {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Length:  {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    if "mean_velocity" in results:
        vel = results["mean_velocity"]
        print(f"Mean Velocity: [{vel[0]:.3f}, {vel[1]:.3f}] m/s")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained H1 walking policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment",
    )
    parser.add_argument(
        "--gym-wrapper",
        action="store_true",
        help="Use Gymnasium wrapper for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    args = parser.parse_args()
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    
    import pickle
    with open(args.checkpoint, "rb") as f:
        checkpoint = pickle.load(f)
    
    config = EnvConfig(num_envs=1)
    env = UnitreeH1Env(config=config)
    
    agent = PPOAgent(
        obs_dim=env.obs_dim,
        action_dim=env.num_actions,
        config=checkpoint.get("config", PPOConfig()),
        rng=jax.random.PRNGKey(args.seed),
    )
    agent.load(args.checkpoint)
    
    print("Starting evaluation...")
    
    if args.gym_wrapper:
        results = evaluate_with_gym(
            agent=agent,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            render=args.render,
            seed=args.seed,
        )
    else:
        results = evaluate_policy(
            agent=agent,
            env=env,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            render=args.render,
        )
    
    print_results(results)


if __name__ == "__main__":
    main()

