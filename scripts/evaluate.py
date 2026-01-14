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


def record_video(
    agent: PPOAgent,
    output_path: str = "walking_demo.mp4",
    num_episodes: int = 1,
    max_steps: int = 500,
    fps: int = 30,
    seed: int = 0,
    width: int = 640,
    height: int = 480,
) -> dict:
    import imageio
    import mujoco
    from mujoco import mjx
    from pathlib import Path
    
    config = EnvConfig(num_envs=1)
    env = UnitreeH1Env(config=config)
    
    scene_path = Path(__file__).parent.parent / "assets" / "unitree_h1" / "scene.xml"
    scene_model = mujoco.MjModel.from_xml_path(str(scene_path))
    scene_model.opt.timestep = config.dt
    scene_model.vis.global_.offwidth = max(width, scene_model.vis.global_.offwidth)
    scene_model.vis.global_.offheight = max(height, scene_model.vis.global_.offheight)
    
    mjx_scene_model = mjx.put_model(scene_model)
    
    renderer = mujoco.Renderer(scene_model, height=height, width=width)
    mj_data = mujoco.MjData(scene_model)
    
    rng = jax.random.PRNGKey(seed)
    frames = []
    episode_rewards = []
    episode_lengths = []
    
    print(f"Recording {num_episodes} episode(s) to {output_path}...")
    
    for ep in range(num_episodes):
        rng, reset_key = jax.random.split(rng)
        
        mjx_data = mjx.make_data(mjx_scene_model)
        qpos = jnp.array(scene_model.qpos0)
        qpos = qpos.at[2].set(0.98)
        mjx_data = mjx_data.replace(qpos=qpos, qvel=jnp.zeros_like(mjx_data.qvel))
        mjx_data = mjx.forward(mjx_scene_model, mjx_data)
        
        command = jnp.array([[0.5, 0.0, 0.0]])
        prev_action = jnp.zeros((1, env.num_actions))
        
        ep_reward = 0.0
        ep_length = 0
        
        for step in range(max_steps):
            obs = env._compute_obs(
                jax.tree.map(lambda x: x[None, ...], mjx_data),
                command
            )
            
            action, _, _, _ = agent.get_action(agent.state, obs, deterministic=True)
            
            scaled_action = action[0] * config.action_scale
            mjx_data = mjx_data.replace(ctrl=scaled_action)
            for _ in range(config.control_decimation):
                mjx_data = mjx.step(mjx_scene_model, mjx_data)
            
            ep_length += 1
            
            mj_data.qpos[:] = np.array(mjx_data.qpos)
            mj_data.qvel[:] = np.array(mjx_data.qvel)
            mujoco.mj_forward(scene_model, mj_data)
            
            torso_id = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
            base_pos = mj_data.xpos[torso_id]
            
            if step == 0:
                print(f"  Initial pos: {base_pos}")
            if step == max_steps - 1:
                print(f"  Final pos: {base_pos}")
            
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.distance = 3.0
            camera.azimuth = 45 + step * 0.5
            camera.elevation = -15
            camera.lookat[:] = [base_pos[0], base_pos[1], max(base_pos[2], 0.3)]
            
            renderer.update_scene(mj_data, camera=camera)
            frame = renderer.render()
            frames.append(frame.copy())
            
            if step % 50 == 0:
                print(f"  Episode {ep + 1}, Step {step}/{max_steps}, height={base_pos[2]:.2f}")
            
            if base_pos[2] < -0.5:
                print(f"  Robot fell at step {step}")
                break
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        print(f"Episode {ep + 1}: Length = {ep_length}")
    
    print(f"Saving video with {len(frames)} frames at {fps} FPS...")
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Video saved to {output_path}")
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "num_frames": len(frames),
        "output_path": output_path,
    }


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
    parser.add_argument(
        "--record-video",
        type=str,
        default=None,
        help="Path to save video (e.g., demo.mp4)",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=30,
        help="Video frames per second",
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=1280,
        help="Video width in pixels",
    )
    parser.add_argument(
        "--video-height",
        type=int,
        default=720,
        help="Video height in pixels",
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
    
    if args.record_video:
        results = record_video(
            agent=agent,
            output_path=args.record_video,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            fps=args.video_fps,
            seed=args.seed,
            width=args.video_width,
            height=args.video_height,
        )
        print_results(results)
        return
    
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

