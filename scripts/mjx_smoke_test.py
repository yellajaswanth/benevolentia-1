#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from physics_ai.config import load_runtime_config
from physics_ai.envs.h1_shared import build_default_qpos, default_asset_path, reset_qpos


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal MJX smoke test for the H1 asset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/h1_walking.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--run-step",
        action="store_true",
        help="Run one mjx.step after mjx.forward",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save smoke-test results as JSON",
    )
    args = parser.parse_args()

    start_time = time.perf_counter()
    results: dict[str, object] = {
        "status": "running",
        "config_path": args.config,
        "run_step": args.run_step,
        "timings": {},
    }

    def mark(stage: str) -> None:
        elapsed = time.perf_counter() - start_time
        results["timings"][stage] = elapsed
        print(f"[{elapsed:8.2f}s] {stage}", flush=True)

    try:
        mark("config_load:start")
        runtime_config = load_runtime_config(args.config)
        asset_path = default_asset_path()
        results["asset_path"] = str(asset_path)
        mark("config_load:done")

        devices = [str(device) for device in jax.devices()]
        results["devices"] = devices
        mark("jax_devices:done")

        mark("mujoco_model_load:start")
        mj_model = mujoco.MjModel.from_xml_path(str(asset_path))
        mj_model.opt.timestep = runtime_config.env.dt
        results["model"] = {
            "nq": int(mj_model.nq),
            "nv": int(mj_model.nv),
            "nu": int(mj_model.nu),
            "ngeom": int(mj_model.ngeom),
        }
        mark("mujoco_model_load:done")

        mark("default_pose_build:start")
        default_qpos = build_default_qpos(mj_model, runtime_config.env.default_joint_angles)
        qpos = reset_qpos(default_qpos, runtime_config.env.initial_height)
        mark("default_pose_build:done")

        mark("mjx_put_model:start")
        mjx_model = mjx.put_model(mj_model)
        mark("mjx_put_model:done")

        mark("mjx_make_data:start")
        data = mjx.make_data(mjx_model)
        data = data.replace(qpos=qpos, qvel=jnp.zeros_like(data.qvel))
        mark("mjx_make_data:done")

        mark("mjx_forward:start")
        data = mjx.forward(mjx_model, data)
        # Force device synchronization so timings reflect actual execution.
        jax.block_until_ready(data.qpos)
        mark("mjx_forward:done")

        if args.run_step:
            mark("mjx_step:start")
            data = mjx.step(mjx_model, data)
            jax.block_until_ready(data.qpos)
            mark("mjx_step:done")

        results["final_state"] = {
            "base_height": float(data.qpos[2]),
            "base_quat": [float(x) for x in data.qpos[3:7]],
            "qvel_norm": float(jnp.linalg.norm(data.qvel)),
        }
        results["status"] = "ok"
    except Exception as exc:  # pragma: no cover - error-path instrumentation
        results["status"] = "error"
        results["failed_stage"] = next(reversed(results["timings"])) if results["timings"] else "unknown"
        results["error"] = f"{type(exc).__name__}: {exc}"
        print(f"Smoke test failed during {results['failed_stage']}: {results['error']}", flush=True)
        raise
    finally:
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, sort_keys=True)
            print(f"Saved smoke-test summary to {output_path}", flush=True)

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
