#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from physics_ai.config import load_runtime_config
from physics_ai.envs.brax_wrapper import create_brax_h1_env
from physics_ai.envs.domain_rand import DomainRandomizer
from physics_ai.envs.h1_env import UnitreeH1Env


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

        mark("unitree_env_create", "start")
        unitree_env = UnitreeH1Env(
            config=runtime_config.env,
            reward_config=runtime_config.reward,
            termination_config=runtime_config.termination,
        )
        mark("unitree_env_create", "done")

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
        mark("unitree_reset", "start")
        unitree_state = unitree_env.reset(rng)
        jax.block_until_ready(unitree_state.obs)
        mark("unitree_reset", "done")

        mark("brax_reset", "start")
        brax_state = brax_env.reset(rng)
        jax.block_until_ready(brax_state.obs)
        mark("brax_reset", "done")

        zero_command = jnp.array([0.0, 0.0, 0.0])
        mark("command_injection", "start")
        unitree_obs = unitree_env._compute_obs(unitree_state.mjx_data, zero_command[None, :])
        unitree_state = unitree_state._replace(obs=unitree_obs, command=zero_command[None, :])
        brax_state = brax_env.set_command(brax_state, zero_command)
        mark("command_injection", "done")

        mark("reset_parity", "start")
        reset_qpos_diff = float(
            np.max(np.abs(np.array(unitree_state.mjx_data.qpos[0]) - np.array(brax_state.pipeline_state.qpos)))
        )
        reset_obs_diff = float(
            np.max(np.abs(np.array(unitree_state.obs[0]) - np.array(brax_state.obs)))
        )
        mark("reset_parity", "done")

        mark("fixed_action_step", "start")
        fixed_action = jnp.array([0.1] + [0.0] * (brax_env.action_size - 1))
        stepped_state = brax_env.step(brax_state, fixed_action)
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

        mark("domain_randomization_sample", "start")
        domain_randomizer = DomainRandomizer(
            config=runtime_config.domain_randomization,
            num_envs=runtime_config.env.num_envs,
        )
        dr_sample = domain_randomizer.sample(rng)
        dr_stats: dict[str, Any] = {}
        for field_name in dr_sample.__dataclass_fields__:
            value = getattr(dr_sample, field_name)
            if value is not None:
                dr_stats[field_name] = {
                    "shape": list(value.shape),
                    "mean": float(jnp.mean(value)),
                    "std": float(jnp.std(value)),
                }
        mark("domain_randomization_sample", "done")

        mark("summary_build", "start")
        summary = {
            "seed": seed,
            "phases": {
                "last_completed": phase_state["last_completed"],
                "elapsed_seconds": time.perf_counter() - start_time,
            },
            "reset_parity": {
                "max_qpos_abs_diff": reset_qpos_diff,
                "max_obs_abs_diff": reset_obs_diff,
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
                "obs_dim_unitree": unitree_env.obs_dim,
                "obs_dim_brax": brax_env.observation_size,
            },
            "actuators": brax_env.diagnostics_summary()["actuators"],
            "domain_randomization": dr_stats,
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
