#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import yaml
from brax.training.agents.ppo.train import train as ppo_train
from brax.training.agents.ppo import networks as ppo_networks

from physics_ai.envs.brax_wrapper import BraxH1EnvWrapper, create_brax_h1_env
from physics_ai.envs.h1_env import EnvConfig
from physics_ai.envs.domain_rand import DomainRandomizationConfig
from physics_ai.rewards.walking import RewardConfig


def load_config(config_path: str) -> tuple[EnvConfig, dict, DomainRandomizationConfig, RewardConfig]:
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
    
    reward_cfg = cfg.get("rewards", {})
    reward_config = RewardConfig(
        reward_scaling=reward_cfg.get("reward_scaling", 0.1),
        velocity_tracking_weight=reward_cfg.get("velocity_tracking", {}).get("weight", 1.0),
        velocity_tracking_scale=reward_cfg.get("velocity_tracking", {}).get("exp_scale", 0.25),
        yaw_rate_weight=reward_cfg.get("yaw_rate_tracking", {}).get("weight", 0.5),
        yaw_rate_scale=reward_cfg.get("yaw_rate_tracking", {}).get("exp_scale", 0.25),
        upright_weight=reward_cfg.get("upright", {}).get("weight", 0.2),
        height_weight=reward_cfg.get("height", {}).get("weight", 0.1),
        target_height=reward_cfg.get("height", {}).get("target_height", 0.98),
        energy_weight=reward_cfg.get("energy", {}).get("weight", -0.001),
        smoothness_weight=reward_cfg.get("smoothness", {}).get("weight", -0.01),
        alive_bonus=reward_cfg.get("alive", {}).get("weight", 1.0),
        termination_penalty=reward_cfg.get("termination", {}).get("weight", -10.0),
    )
    
    brax_ppo_cfg = cfg.get("brax_ppo", {})
    ppo_cfg = cfg.get("ppo", {})
    train_cfg = cfg.get("training", {})
    
    brax_config = {
        "num_timesteps": train_cfg.get("total_timesteps", 400_000_000),
        "num_evals": train_cfg.get("num_evals", 100),
        "episode_length": env_config.episode_length,
        "num_envs": env_config.num_envs,
        "learning_rate": brax_ppo_cfg.get("learning_rate", ppo_cfg.get("learning_rate", 3e-4)),
        "entropy_cost": brax_ppo_cfg.get("entropy_cost", 0.001),
        "discounting": brax_ppo_cfg.get("discounting", ppo_cfg.get("gamma", 0.99)),
        "unroll_length": brax_ppo_cfg.get("unroll_length", train_cfg.get("rollout_length", 32)),
        "batch_size": brax_ppo_cfg.get("batch_size", env_config.num_envs * train_cfg.get("rollout_length", 32)),
        "num_minibatches": brax_ppo_cfg.get("num_minibatches", ppo_cfg.get("num_minibatches", 32)),
        "num_updates_per_batch": brax_ppo_cfg.get("num_updates_per_batch", ppo_cfg.get("num_epochs", 4)),
        "normalize_observations": brax_ppo_cfg.get("normalize_observations", True),
        "reward_scaling": brax_ppo_cfg.get("reward_scaling", 1.0),
        "clipping_epsilon": brax_ppo_cfg.get("clipping_epsilon", ppo_cfg.get("clip_ratio", 0.2)),
        "gae_lambda": brax_ppo_cfg.get("gae_lambda", ppo_cfg.get("gae_lambda", 0.95)),
        "seed": train_cfg.get("seed", 42),
    }
    
    return env_config, brax_config, dr_config, reward_config


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
    env_config, brax_config, dr_config, reward_config = load_config(config_path)
    
    if seed is not None:
        brax_config["seed"] = seed
    if num_envs is not None:
        env_config = EnvConfig(
            num_envs=num_envs,
            episode_length=env_config.episode_length,
            dt=env_config.dt,
            control_decimation=env_config.control_decimation,
            vx_range=env_config.vx_range,
            vy_range=env_config.vy_range,
            vyaw_range=env_config.vyaw_range,
            command_resample_time=env_config.command_resample_time,
        )
        brax_config["num_envs"] = num_envs
        brax_config["batch_size"] = num_envs * brax_config["unroll_length"]
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
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
        
        if len(eval_steps) % 10 == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_{num_steps:012d}.pkl"
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
    
    inference_fn = make_inference_fn(params, deterministic=True)
    inference_path = os.path.join(checkpoint_dir, "brax_inference.pkl")
    with open(inference_path, "wb") as f:
        pickle.dump({
            "params": params,
            "obs_size": env.observation_size,
            "action_size": env.action_size,
        }, f)
    print(f"Saved inference data to {inference_path}")
    
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
