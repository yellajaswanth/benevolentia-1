#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import yaml
from brax.training.agents.ppo import networks as ppo_networks

from physics_ai.config import RuntimeConfig, load_runtime_config
from physics_ai.envs.brax_wrapper import create_brax_h1_env
from physics_ai.utils.jax_utils import quat_rotate_inverse


def load_policy_bundle(checkpoint_path: str) -> dict[str, Any]:
    with open(checkpoint_path, "rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict) or "params" not in bundle:
        raise ValueError(
            "Expected a canonical Brax policy bundle with a 'params' field. "
            "Use checkpoints/brax_policy.pkl from scripts/train_brax.py."
        )
    return bundle


def runtime_config_from_bundle(bundle: dict[str, Any], checkpoint_path: str) -> RuntimeConfig:
    resolved = bundle.get("resolved_config")
    if resolved:
        temp_path = Path(checkpoint_path).with_name(".tmp_resolved_config_for_eval.yaml")
        with open(temp_path, "w") as f:
            yaml.safe_dump(resolved, f, sort_keys=False)
        try:
            return load_runtime_config(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    sibling = Path(checkpoint_path).with_name("resolved_config.yaml")
    if sibling.exists():
        return load_runtime_config(sibling)

    return load_runtime_config("configs/h1_walking.yaml")


def create_inference_fn(bundle: dict[str, Any]):
    ppo_network = ppo_networks.make_ppo_networks(bundle["obs_size"], bundle["action_size"])
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    return make_inference_fn(bundle["params"], deterministic=True)


def compute_local_velocity(pipeline_state) -> np.ndarray:
    base_quat = pipeline_state.qpos[3:7]
    base_lin_vel = pipeline_state.qvel[:3]
    base_lin_vel_local = quat_rotate_inverse(base_quat, base_lin_vel)
    return np.array(base_lin_vel_local)


def evaluate_scenario(
    env,
    inference_fn,
    seed: int,
    command: jnp.ndarray,
    max_steps: int,
) -> dict[str, Any]:
    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)
    state = env.set_command(state, command)

    reward_breakdown_sum = {
        "velocity_tracking": 0.0,
        "yaw_tracking": 0.0,
        "upright": 0.0,
        "height": 0.0,
        "energy": 0.0,
        "smoothness": 0.0,
        "alive": 0.0,
        "termination_penalty": 0.0,
    }
    achieved_local_velocities = []
    start_x = float(state.pipeline_state.qpos[0])
    final_x = start_x
    fall_count = 0
    steps = 0

    for _ in range(max_steps):
        rng, act_rng = jax.random.split(rng)
        prev_action = state.info["prev_action"]
        command_before_step = state.info["command"]
        action, _ = inference_fn(state.obs, act_rng)
        action = jnp.asarray(action)
        state = env.step(state, action)
        steps += 1
        final_x = float(state.pipeline_state.qpos[0])

        breakdown = env._compute_reward_components(
            mjx_data=state.pipeline_state,
            action=action,
            prev_action=prev_action,
            command=command_before_step,
            done=state.done > 0,
        )
        for key in reward_breakdown_sum:
            reward_breakdown_sum[key] += float(breakdown[key])

        achieved_local_velocities.append(compute_local_velocity(state.pipeline_state))

        if float(state.done) > 0:
            fall_count = 1
            break

    mean_local_velocity = (
        np.mean(np.stack(achieved_local_velocities, axis=0), axis=0)
        if achieved_local_velocities
        else np.zeros(3)
    )
    denom = max(steps, 1)

    return {
        "seed": seed,
        "command": np.array(command).tolist(),
        "episode_length": steps,
        "fell": bool(fall_count),
        "fall_rate": float(fall_count),
        "distance_x": final_x - start_x,
        "mean_local_velocity": mean_local_velocity.tolist(),
        "reward_breakdown": {key: value / denom for key, value in reward_breakdown_sum.items()},
    }


def evaluate_bundle(
    bundle: dict[str, Any],
    runtime_config: RuntimeConfig,
    max_steps: int | None = None,
) -> dict[str, Any]:
    inference_fn = create_inference_fn(bundle)
    env = create_brax_h1_env(
        env_config=runtime_config.env,
        reward_config=runtime_config.reward,
        dr_config=runtime_config.domain_randomization,
        termination_config=runtime_config.termination,
    )

    scenarios = []
    max_eval_steps = max_steps or runtime_config.evaluation.max_steps
    for scenario in runtime_config.evaluation.fixed_commands:
        seed_results = []
        command = jnp.array(scenario.command)
        for seed in runtime_config.evaluation.seeds:
            seed_results.append(
                evaluate_scenario(
                    env=env,
                    inference_fn=inference_fn,
                    seed=seed,
                    command=command,
                    max_steps=max_eval_steps,
                )
            )

        scenario_summary = {
            "name": scenario.name,
            "command": list(scenario.command),
            "mean_episode_length": float(np.mean([r["episode_length"] for r in seed_results])),
            "mean_fall_rate": float(np.mean([r["fall_rate"] for r in seed_results])),
            "mean_distance_x": float(np.mean([r["distance_x"] for r in seed_results])),
            "mean_local_velocity": np.mean(
                np.array([r["mean_local_velocity"] for r in seed_results]),
                axis=0,
            ).tolist(),
            "mean_reward_breakdown": {
                key: float(np.mean([r["reward_breakdown"][key] for r in seed_results]))
                for key in seed_results[0]["reward_breakdown"]
            },
            "seed_results": seed_results,
        }
        scenarios.append(scenario_summary)

    return {
        "checkpoint_seed": bundle.get("seed"),
        "evaluation_seeds": list(runtime_config.evaluation.seeds),
        "scenarios": scenarios,
    }


def record_video(
    bundle: dict[str, Any],
    runtime_config: RuntimeConfig,
    output_path: str,
    scenario_name: str,
    seed: int,
    max_steps: int | None = None,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
) -> None:
    inference_fn = create_inference_fn(bundle)
    env = create_brax_h1_env(
        env_config=runtime_config.env,
        reward_config=runtime_config.reward,
        dr_config=runtime_config.domain_randomization,
        termination_config=runtime_config.termination,
    )
    scenario = next((item for item in runtime_config.evaluation.fixed_commands if item.name == scenario_name), None)
    if scenario is None:
        available = [item.name for item in runtime_config.evaluation.fixed_commands]
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available scenarios: {available}")

    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)
    state = env.set_command(state, jnp.array(scenario.command))

    scene_path = Path(__file__).parent.parent / "assets" / "unitree_h1" / "scene.xml"
    scene_model = mujoco.MjModel.from_xml_path(str(scene_path))
    scene_model.opt.timestep = runtime_config.env.dt
    scene_model.vis.global_.offwidth = max(width, scene_model.vis.global_.offwidth)
    scene_model.vis.global_.offheight = max(height, scene_model.vis.global_.offheight)

    renderer = mujoco.Renderer(scene_model, height=height, width=width)
    mj_data = mujoco.MjData(scene_model)
    frames = []

    for _ in range(max_steps or runtime_config.evaluation.max_steps):
        rng, act_rng = jax.random.split(rng)
        action, _ = inference_fn(state.obs, act_rng)
        action = jnp.asarray(action)
        state = env.step(state, action)

        mj_data.qpos[:] = np.array(state.pipeline_state.qpos)
        mj_data.qvel[:] = np.array(state.pipeline_state.qvel)
        mujoco.mj_forward(scene_model, mj_data)

        torso_id = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        base_pos = mj_data.xpos[torso_id]

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 3.0
        camera.azimuth = 45
        camera.elevation = -15
        camera.lookat[:] = [base_pos[0], base_pos[1], max(base_pos[2], 0.3)]

        renderer.update_scene(mj_data, camera=camera)
        frames.append(renderer.render().copy())

        if float(state.done) > 0:
            break

    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Saved video to {output_path}")


def print_results(results: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("Canonical Brax Evaluation Results")
    print("=" * 60)
    for scenario in results["scenarios"]:
        print(f"Scenario: {scenario['name']} command={scenario['command']}")
        print(
            f"  Episode length: {scenario['mean_episode_length']:.1f} | "
            f"Fall rate: {scenario['mean_fall_rate']:.2f} | "
            f"Distance x: {scenario['mean_distance_x']:.3f}"
        )
        print(
            f"  Mean local velocity: "
            f"{[round(v, 3) for v in scenario['mean_local_velocity']]}"
        )
        print(
            f"  Reward breakdown: "
            f"{json.dumps(scenario['mean_reward_breakdown'], sort_keys=True)}"
        )
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate canonical Brax PPO H1 policy bundle")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to canonical Brax policy bundle (e.g., checkpoints/brax_policy.pkl)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override maximum steps per evaluation scenario",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save the evaluation summary as JSON",
    )
    parser.add_argument(
        "--record-video",
        type=str,
        default=None,
        help="Optional path to save a canonical Brax rollout video",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="forward_fast",
        help="Scenario name to use when recording video",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed to use when recording video",
    )
    args = parser.parse_args()

    bundle = load_policy_bundle(args.checkpoint)
    runtime_config = runtime_config_from_bundle(bundle, args.checkpoint)
    if args.record_video:
        record_video(
            bundle=bundle,
            runtime_config=runtime_config,
            output_path=args.record_video,
            scenario_name=args.scenario,
            seed=args.seed,
            max_steps=args.max_steps,
        )
        return

    results = evaluate_bundle(bundle, runtime_config, max_steps=args.max_steps)
    print_results(results)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print(f"Saved evaluation summary to {args.output_json}")


if __name__ == "__main__":
    main()
