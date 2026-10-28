#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from physics_ai.config import load_runtime_config
from physics_ai.envs.brax_wrapper import create_brax_h1_env


def main():
    parser = argparse.ArgumentParser(description="Run deterministic H1 locomotion diagnostics")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/h1_walking.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save diagnostics summary as JSON",
    )
    parser.add_argument(
        "--verbose-phases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print timing checkpoints before and after major diagnostics phases",
    )
    args = parser.parse_args()

    start_time = time.perf_counter()
    phase_state: dict[str, str | None] = {"current": None, "last_completed": None}

    def mark(phase: str, event: str) -> None:
        phase_state["current"] = phase if event == "start" else phase_state["current"]
        if event == "done":
            phase_state["last_completed"] = phase
            phase_state["current"] = None
        if args.verbose_phases:
            elapsed = time.perf_counter() - start_time
            print(f"[{elapsed:8.2f}s] {phase}:{event}", flush=True)

    try:
        mark("config_load", "start")
        runtime_config = load_runtime_config(args.config)
        mark("config_load", "done")

        mark("brax_env_create", "start")
        brax_env = create_brax_h1_env(
            env_config=runtime_config.env,
            reward_config=runtime_config.reward,
            dr_config=runtime_config.domain_randomization,
            termination_config=runtime_config.termination,
        )
        mark("brax_env_create", "done")

        seed = 0
        rng = jax.random.PRNGKey(seed)
        mark("brax_reset", "start")
        brax_state = brax_env.reset(rng)
        jax.block_until_ready(brax_state.obs)
        mark("brax_reset", "done")

        zero_command = jnp.array([0.0, 0.0, 0.0])
        mark("command_injection", "start")
        brax_state = brax_env.set_command(brax_state, zero_command)
        mark("command_injection", "done")

        step_fn = jax.jit(brax_env.step)

        mark("standing_settle", "start")
        settle_state = brax_state
        zero_action = jnp.zeros(brax_env.action_size)
        for _ in range(50):
            settle_state = step_fn(settle_state, zero_action)
        settle_contact = np.array(brax_env.foot_contact(settle_state.pipeline_state)).tolist()
        mark("standing_settle", "done")

        mark("foot_lift", "start")
        # Lift the left leg by bending hip and knee; the left foot should leave the floor.
        lift_action = jnp.zeros(brax_env.action_size)
        for joint, offset in (("left_hip_pitch", -0.6), ("left_knee", 1.2)):
            joint_id = mujoco.mj_name2id(brax_env._mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            actuator = int(np.where(brax_env._mj_model.actuator_trnid[:, 0] == joint_id)[0][0])
            lift_action = lift_action.at[actuator].set(offset / runtime_config.env.action_scale)
        lift_state = brax_state
        for _ in range(10):
            lift_state = step_fn(lift_state, lift_action)
        lift_contact = np.array(brax_env.foot_contact(lift_state.pipeline_state)).tolist()
        mark("foot_lift", "done")

        mark("fixed_action_step", "start")
        fixed_action = jnp.array([0.1] + [0.0] * (brax_env.action_size - 1))
        stepped_state = step_fn(brax_state, fixed_action)
        jax.block_until_ready(stepped_state.obs)
        qpos_delta = np.array(stepped_state.pipeline_state.qpos - brax_state.pipeline_state.qpos)
        qvel_delta = np.array(stepped_state.pipeline_state.qvel - brax_state.pipeline_state.qvel)
        mark("fixed_action_step", "done")

        mark("termination_check_low", "start")
        low_state = brax_state.replace(
            pipeline_state=brax_state.pipeline_state.replace(
                qpos=brax_state.pipeline_state.qpos.at[2].set(runtime_config.termination.min_height - 0.05)
            )
        )
        low_height_triggers_done = bool(brax_env._compute_termination(low_state.pipeline_state))
        mark("termination_check_low", "done")

        mark("termination_check_tilt", "start")
        tilted_qpos = brax_state.pipeline_state.qpos.at[3:7].set(jnp.array([0.9659, 0.2588, 0.0, 0.0]))
        tilt_state = brax_state.replace(
            pipeline_state=brax_state.pipeline_state.replace(qpos=tilted_qpos)
        )
        tilt_triggers_done = bool(brax_env._compute_termination(tilt_state.pipeline_state))
        mark("termination_check_tilt", "done")

        mark("summary_build", "start")
        summary = {
            "seed": seed,
            "phases": {
                "last_completed": phase_state["last_completed"],
                "elapsed_seconds": time.perf_counter() - start_time,
            },
            "feet": {
                "contact_at_reset": np.array(brax_env.foot_contact(brax_state.pipeline_state)).tolist(),
                "contact_after_1s_zero_action": settle_contact,
                "base_height_after_1s": float(settle_state.pipeline_state.qpos[2]),
                "contact_after_left_leg_lift": lift_contact,
            },
            "action_semantics": {
                "qpos_delta_l2": float(np.linalg.norm(qpos_delta)),
                "qvel_delta_l2": float(np.linalg.norm(qvel_delta)),
                "default_ctrl_preview": brax_env.diagnostics_summary()["default_ctrl"][:5],
            },
            "termination": {
                "low_height_triggers_done": low_height_triggers_done,
                "tilt_triggers_done": tilt_triggers_done,
            },
            "observation": {
                "obs_dim": brax_env.observation_size,
                "obs_version": brax_env.obs_version,
            },
            "initial_height": brax_env.diagnostics_summary()["initial_height"],
            "actuators": brax_env.diagnostics_summary()["actuators"],
        }
        mark("summary_build", "done")

        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.output_json:
            mark("summary_write", "start")
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2, sort_keys=True)
            print(f"Saved diagnostics summary to {output_path}")
            mark("summary_write", "done")
    except Exception:
        elapsed = time.perf_counter() - start_time
        print(
            json.dumps(
                {
                    "status": "error",
                    "current_phase": phase_state["current"],
                    "last_completed_phase": phase_state["last_completed"],
                    "elapsed_seconds": elapsed,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
