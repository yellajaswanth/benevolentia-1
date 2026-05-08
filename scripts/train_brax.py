#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

from brax.training.agents.ppo.train import train as ppo_train

from physics_ai.config import (
    load_runtime_config,
    print_runtime_summary,
    runtime_config_to_dict,
    write_runtime_metadata,
)
from physics_ai.envs.brax_wrapper import create_brax_h1_env


def save_params(params, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(params, f)
    print(f"Saved params to {path}")


def train(
    config_path: str,
    checkpoint_dir: str = "checkpoints",
    seed: int | None = None,
    num_envs: int | None = None,
):
    runtime_config = load_runtime_config(config_path)
    env_config = runtime_config.env
    reward_config = runtime_config.reward
    dr_config = runtime_config.domain_randomization
    termination_config = runtime_config.termination
    brax_config = dict(runtime_config.brax_ppo)
    
    if seed is not None:
        brax_config["seed"] = seed
        runtime_config.brax_ppo["seed"] = seed
    if num_envs is not None:
        env_config.num_envs = num_envs
        brax_config["num_envs"] = num_envs
        brax_config["batch_size"] = num_envs * brax_config["unroll_length"]
        runtime_config.brax_ppo["num_envs"] = num_envs
        runtime_config.brax_ppo["batch_size"] = brax_config["batch_size"]
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    print_runtime_summary(runtime_config)
    write_runtime_metadata(
        output_dir=checkpoint_dir,
        config_path=config_path,
        runtime_config=runtime_config,
        extra_metadata={"seed": brax_config["seed"]},
    )
    
    print("=" * 60)
    print("Brax PPO Training for Unitree H1")
    print("=" * 60)
    print(f"\nEnvironment Configuration:")
    print(f"  Num environments: {env_config.num_envs}")
    print(f"  Episode length: {env_config.episode_length}")
    print(f"  Control dt: {env_config.dt * env_config.control_decimation:.4f}s")
    print(f"\nBrax PPO Configuration:")
    for key, value in brax_config.items():
        print(f"  {key}: {value}")
    print()
    
    print("Creating environment...")
    env = create_brax_h1_env(
        env_config=env_config,
        reward_config=reward_config,
        dr_config=dr_config,
        termination_config=termination_config,
    )
    print(f"  Observation size: {env.observation_size}")
    print(f"  Action size: {env.action_size}")
    
    times = [datetime.now()]
    eval_rewards = []
    eval_steps = []
    
    def progress_fn(num_steps: int, metrics: dict):
        times.append(datetime.now())
        
        eval_reward = metrics.get("eval/episode_reward", 0.0)
        eval_reward_std = metrics.get("eval/episode_reward_std", 0.0)
        
        eval_rewards.append(eval_reward)
        eval_steps.append(num_steps)
        
        elapsed = (times[-1] - times[0]).total_seconds()
        fps = num_steps / elapsed if elapsed > 0 else 0
        
        print(
            f"Steps: {num_steps:12,} | "
            f"FPS: {fps:10,.0f} | "
            f"Reward: {eval_reward:8.3f} +/- {eval_reward_std:6.3f} | "
            f"Time: {elapsed/60:6.1f}min"
        )
        
    
    print("\nStarting training...")
    print("-" * 60)
    
    start_time = time.time()
    
    make_inference_fn, params, metrics = ppo_train(
        environment=env,
        progress_fn=progress_fn,
        **brax_config,
    )
    
    total_time = time.time() - start_time
    
    print("-" * 60)
    print(f"\nTraining completed!")
    print(f"  Total time: {total_time / 60:.2f} minutes ({total_time / 3600:.2f} hours)")
    print(f"  Final reward: {eval_rewards[-1] if eval_rewards else 'N/A':.3f}")
    print(f"  Average FPS: {brax_config['num_timesteps'] / total_time:,.0f}")
    
    final_path = os.path.join(checkpoint_dir, "brax_final.pkl")
    save_params(params, final_path)
    
    policy_bundle_path = os.path.join(checkpoint_dir, "brax_policy.pkl")
    with open(policy_bundle_path, "wb") as f:
        pickle.dump({
            "params": params,
            "obs_size": env.observation_size,
            "action_size": env.action_size,
            "resolved_config": runtime_config_to_dict(runtime_config),
            "seed": brax_config["seed"],
        }, f)
    print(f"Saved canonical policy bundle to {policy_bundle_path}")
    
    return make_inference_fn, params, metrics


def main():
    parser = argparse.ArgumentParser(description="Train H1 walking policy with Brax PPO")
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
    
    if not config_path.exists():
        config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return
    
    train(
        config_path=str(config_path),
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        num_envs=args.num_envs,
    )


if __name__ == "__main__":
    main()
