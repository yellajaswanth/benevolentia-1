from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from physics_ai.envs.domain_rand import DomainRandomizationConfig
from physics_ai.envs.h1_structs import EnvConfig
from physics_ai.rewards.walking import RewardConfig


def default_joint_angles() -> dict[str, float]:
    return {
        "left_hip_yaw": 0.0,
        "left_hip_roll": 0.0,
        "left_hip_pitch": -0.4,
        "left_knee": 0.8,
        "left_ankle": -0.4,
        "right_hip_yaw": 0.0,
        "right_hip_roll": 0.0,
        "right_hip_pitch": -0.4,
        "right_knee": 0.8,
        "right_ankle": -0.4,
        "torso": 0.0,
        "left_shoulder_pitch": 0.0,
        "left_shoulder_roll": 0.0,
        "left_shoulder_yaw": 0.0,
        "left_elbow": 0.0,
        "right_shoulder_pitch": 0.0,
        "right_shoulder_roll": 0.0,
        "right_shoulder_yaw": 0.0,
        "right_elbow": 0.0,
    }


@dataclass
class TerminationConfig:
    min_height: float = 0.3
    max_pitch: float = 0.5
    max_roll: float = 0.5


@dataclass
class EvaluationScenario:
    name: str
    command: tuple[float, float, float]


@dataclass
class EvaluationConfig:
    seeds: tuple[int, ...] = (0, 1, 2)
    max_steps: int = 1000
    fixed_commands: tuple[EvaluationScenario, ...] = (
        EvaluationScenario(name="stand", command=(0.0, 0.0, 0.0)),
        EvaluationScenario(name="forward_slow", command=(0.25, 0.0, 0.0)),
        EvaluationScenario(name="forward_fast", command=(0.5, 0.0, 0.0)),
    )


@dataclass
class RuntimeConfig:
    env: EnvConfig
    reward: RewardConfig
    domain_randomization: DomainRandomizationConfig
    termination: TerminationConfig
    brax_ppo: dict[str, Any]
    evaluation: EvaluationConfig
    raw_config: dict[str, Any] = field(default_factory=dict)


def load_runtime_config(config_path: str | Path) -> RuntimeConfig:
    path = Path(config_path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    robot_cfg = cfg.get("robot", {})
    env_cfg = cfg.get("env", {})
    commands_cfg = cfg.get("commands", {})
    rewards_cfg = cfg.get("rewards", {})
    termination_cfg = cfg.get("termination", {})
    dr_cfg = cfg.get("domain_randomization", {})
    train_cfg = cfg.get("training", {})
    ppo_cfg = cfg.get("ppo", {})
    brax_ppo_cfg = cfg.get("brax_ppo", {})
    eval_cfg = cfg.get("evaluation", {})

    env = EnvConfig(
        num_envs=env_cfg.get("num_envs", 4096),
        episode_length=env_cfg.get("episode_length", 1000),
        dt=env_cfg.get("dt", 0.005),
        control_decimation=env_cfg.get("control_decimation", 4),
        action_scale=robot_cfg.get("action_scale", 0.25),
        default_joint_angles=robot_cfg.get("default_joint_angles", default_joint_angles()),
        initial_height=robot_cfg.get("initial_height", 0.88),
        vx_range=tuple(commands_cfg.get("vx_range", [-1.0, 1.0])),
        vy_range=tuple(commands_cfg.get("vy_range", [-0.5, 0.5])),
        vyaw_range=tuple(commands_cfg.get("vyaw_range", [-1.0, 1.0])),
        command_resample_time=commands_cfg.get("resample_time", 10.0),
    )

    reward = RewardConfig(
        reward_scaling=rewards_cfg.get("reward_scaling", 0.1),
        velocity_tracking_weight=rewards_cfg.get("velocity_tracking", {}).get("weight", 1.0),
        velocity_tracking_scale=rewards_cfg.get("velocity_tracking", {}).get("exp_scale", 0.25),
        yaw_rate_weight=rewards_cfg.get("yaw_rate_tracking", {}).get("weight", 0.5),
        yaw_rate_scale=rewards_cfg.get("yaw_rate_tracking", {}).get("exp_scale", 0.25),
        upright_weight=rewards_cfg.get("upright", {}).get("weight", 0.2),
        height_weight=rewards_cfg.get("height", {}).get("weight", 0.1),
        target_height=rewards_cfg.get("height", {}).get("target_height", env.initial_height),
        energy_weight=rewards_cfg.get("energy", {}).get("weight", -0.001),
        smoothness_weight=rewards_cfg.get("smoothness", {}).get("weight", -0.01),
        alive_bonus=rewards_cfg.get("alive", {}).get("weight", 1.0),
        termination_penalty=rewards_cfg.get("termination", {}).get("weight", -10.0),
    )

    termination = TerminationConfig(
        min_height=termination_cfg.get("min_height", 0.3),
        max_pitch=termination_cfg.get("max_pitch", 0.5),
        max_roll=termination_cfg.get("max_roll", 0.5),
    )

    domain_randomization = DomainRandomizationConfig(
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

    brax_ppo = {
        "num_timesteps": train_cfg.get("total_timesteps", 400_000_000),
        "num_evals": train_cfg.get("num_evals", 100),
        "episode_length": env.episode_length,
        "num_envs": env.num_envs,
        "learning_rate": brax_ppo_cfg.get("learning_rate", ppo_cfg.get("learning_rate", 3e-4)),
        "entropy_cost": brax_ppo_cfg.get("entropy_cost", 0.001),
        "discounting": brax_ppo_cfg.get("discounting", ppo_cfg.get("gamma", 0.99)),
        "unroll_length": brax_ppo_cfg.get("unroll_length", train_cfg.get("rollout_length", 32)),
        "batch_size": brax_ppo_cfg.get("batch_size", env.num_envs * train_cfg.get("rollout_length", 32)),
        "num_minibatches": brax_ppo_cfg.get("num_minibatches", ppo_cfg.get("num_minibatches", 32)),
        "num_updates_per_batch": brax_ppo_cfg.get("num_updates_per_batch", ppo_cfg.get("num_epochs", 4)),
        "normalize_observations": brax_ppo_cfg.get("normalize_observations", True),
        "reward_scaling": brax_ppo_cfg.get("reward_scaling", 1.0),
        "clipping_epsilon": brax_ppo_cfg.get("clipping_epsilon", ppo_cfg.get("clip_ratio", 0.2)),
        "gae_lambda": brax_ppo_cfg.get("gae_lambda", ppo_cfg.get("gae_lambda", 0.95)),
        "seed": train_cfg.get("seed", 42),
    }

    scenarios = []
    for idx, item in enumerate(eval_cfg.get("fixed_commands", [])):
        scenarios.append(
            EvaluationScenario(
                name=item.get("name", f"scenario_{idx}"),
                command=tuple(item.get("command", [0.0, 0.0, 0.0])),
            )
        )
    evaluation = EvaluationConfig(
        seeds=tuple(eval_cfg.get("seeds", [0, 1, 2])),
        max_steps=eval_cfg.get("max_steps", env.episode_length),
        fixed_commands=tuple(scenarios) if scenarios else EvaluationConfig().fixed_commands,
    )

    return RuntimeConfig(
        env=env,
        reward=reward,
        domain_randomization=domain_randomization,
        termination=termination,
        brax_ppo=brax_ppo,
        evaluation=evaluation,
        raw_config=cfg,
    )


def _to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return _to_plain_data(asdict(value))
    if isinstance(value, dict):
        return {k: _to_plain_data(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_plain_data(v) for v in value]
    if isinstance(value, list):
        return [_to_plain_data(v) for v in value]
    return value


def runtime_config_to_dict(config: RuntimeConfig) -> dict[str, Any]:
    return _to_plain_data(config)


def print_runtime_summary(config: RuntimeConfig) -> None:
    data = runtime_config_to_dict(config)
    print("=" * 60)
    print("Resolved Runtime Configuration")
    print("=" * 60)
    print(json.dumps(data, indent=2, sort_keys=True))
    print("=" * 60)


def current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def write_runtime_metadata(
    output_dir: str | Path,
    config_path: str | Path,
    runtime_config: RuntimeConfig,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    resolved_config_path = output_path / "resolved_config.yaml"
    metadata_path = output_path / "run_metadata.json"

    with open(resolved_config_path, "w") as f:
        yaml.safe_dump(runtime_config_to_dict(runtime_config), f, sort_keys=False)

    metadata = {
        "git_commit": current_git_commit(),
        "config_path": str(config_path),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
