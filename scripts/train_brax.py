#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo.train import train as ppo_train

from physics_ai.config import (
    load_runtime_config,
    print_runtime_summary,
    runtime_config_to_dict,
    write_runtime_metadata,
)
from physics_ai.envs.brax_wrapper import create_brax_h1_env
from physics_ai.utils.jax_compat import install_brax_compat

install_brax_compat()

# Reward terms printed in the progress line; all terms go to progress.jsonl.
PROGRESS_TERMS = ("tracking_lin_vel", "feet_air_time", "contact_match", "termination")


def save_params(params, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(params, f)
    print(f"Saved params to {path}")


def policy_bundle(params, env, runtime_config, seed: int) -> dict:
    return {
        "params": params,
        "obs_size": env.observation_size,
        "action_size": env.action_size,
        "obs_version": env.obs_version,
        "network": runtime_config_to_dict(runtime_config)["network"],
        "resolved_config": runtime_config_to_dict(runtime_config),
        "seed": seed,
    }


def train(
    config_path: str,
    checkpoint_dir: str = "checkpoints",
    seed: int | None = None,
    num_envs: int | None = None,
    num_timesteps: int | None = None,
    allow_cpu: bool = False,
):
    backend = jax.default_backend()
    if backend != "gpu" and not allow_cpu:
        sys.exit(
            f"JAX backend is '{backend}', not 'gpu'. Humanoid training needs a GPU; "
            "pass --allow-cpu only for smoke tests."
        )

    runtime_config = load_runtime_config(config_path)
    env_config = runtime_config.env
    brax_config = dict(runtime_config.brax_ppo)

    if seed is not None:
        brax_config["seed"] = seed
        runtime_config.brax_ppo["seed"] = seed
    if num_envs is not None:
        env_config.num_envs = num_envs
        brax_config["num_envs"] = num_envs
        brax_config["batch_size"] = max(num_envs // brax_config["num_minibatches"], 1)
        runtime_config.brax_ppo["num_envs"] = num_envs
        runtime_config.brax_ppo["batch_size"] = brax_config["batch_size"]
    if num_timesteps is not None:
        brax_config["num_timesteps"] = num_timesteps
        runtime_config.brax_ppo["num_timesteps"] = num_timesteps

    os.makedirs(checkpoint_dir, exist_ok=True)
    print_runtime_summary(runtime_config)
    write_runtime_metadata(
        output_dir=checkpoint_dir,
        config_path=config_path,
        runtime_config=runtime_config,
        extra_metadata={"seed": brax_config["seed"], "jax_devices": [str(d) for d in jax.devices()]},
    )

    print("Creating environment...")
    env = create_brax_h1_env(
        env_config=env_config,
        reward_config=runtime_config.reward,
        dr_config=runtime_config.domain_randomization,
        termination_config=runtime_config.termination,
    )
    print(f"  Observation size: {env.observation_size} (v{env.obs_version})")
    print(f"  Action size: {env.action_size}")
    print(f"  Initial height: {env.diagnostics_summary()['initial_height']:.3f} m")

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=runtime_config.network.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=runtime_config.network.value_hidden_layer_sizes,
    )

    start_time = time.time()
    progress_path = Path(checkpoint_dir) / "progress.jsonl"
    eval_rewards = []

    def progress_fn(num_steps: int, metrics: dict):
        elapsed = time.time() - start_time
        record = {"num_steps": int(num_steps), "elapsed_s": elapsed}
        record.update({k: float(v) for k, v in metrics.items() if k.startswith("eval/")})
        with open(progress_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        eval_reward = record.get("eval/episode_reward", 0.0)
        eval_rewards.append(eval_reward)
        ep_len = record.get("eval/avg_episode_length", 0.0)
        terms = " ".join(
            f"{name}={record.get(f'eval/episode_reward/{name}', 0.0):.1f}" for name in PROGRESS_TERMS
        )
        print(
            f"Steps: {num_steps:12,} | SPS: {num_steps / max(elapsed, 1e-6):9,.0f} | "
            f"Reward: {eval_reward:8.2f} | EpLen: {ep_len:6.1f} | {terms} | "
            f"Time: {elapsed / 60:6.1f}min",
            flush=True,
        )

    def policy_params_fn(current_step, make_policy, params):
        path = os.path.join(checkpoint_dir, f"brax_policy_{int(current_step):012d}.pkl")
        with open(path, "wb") as f:
            pickle.dump(policy_bundle(params, env, runtime_config, brax_config["seed"]), f)

    print("\nStarting training...")
    print("-" * 60)

    make_inference_fn, params, metrics = ppo_train(
        environment=env,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
        network_factory=network_factory,
        **brax_config,
    )

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"\nTraining completed in {total_time / 60:.2f} minutes")
    if eval_rewards:
        print(f"  Final eval reward: {eval_rewards[-1]:.3f}")
    print(f"  Average SPS: {brax_config['num_timesteps'] / total_time:,.0f}")

    save_params(params, os.path.join(checkpoint_dir, "brax_final.pkl"))
    policy_bundle_path = os.path.join(checkpoint_dir, "brax_policy.pkl")
    with open(policy_bundle_path, "wb") as f:
        pickle.dump(policy_bundle(params, env, runtime_config, brax_config["seed"]), f)
    print(f"Saved canonical policy bundle to {policy_bundle_path}")

    return make_inference_fn, params, metrics


def main():
    parser = argparse.ArgumentParser(description="Train H1 walking policy with Brax PPO")
    parser.add_argument("--config", type=str, default="configs/h1_walking.yaml", help="Path to config file")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--num-envs", type=int, default=None, help="Override number of parallel environments")
    parser.add_argument("--num-timesteps", type=int, default=None, help="Override total environment steps")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow non-GPU backends (smoke tests only)")
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent / args.config
    if not config_path.exists():
        config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return

    print(f"Run started {datetime.now().isoformat(timespec='seconds')}")
    train(
        config_path=str(config_path),
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        num_envs=args.num_envs,
        num_timesteps=args.num_timesteps,
        allow_cpu=args.allow_cpu,
    )


if __name__ == "__main__":
    main()
