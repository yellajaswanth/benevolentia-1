"""Environment integrity checks: floor, feet, gait clock, observation layout.

Slow on CPU (each jit of env.step takes ~30 s), so steps share one module-level env.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from physics_ai.config import load_runtime_config
from physics_ai.envs.brax_wrapper import create_brax_h1_env
from physics_ai.envs.domain_rand import DomainRandomizationConfig
from physics_ai.envs.h1_shared import OBS_VERSION, compute_termination, gait_frequency, observation_size


@pytest.fixture(scope="module")
def runtime_config():
    return load_runtime_config("configs/h1_walking.yaml")


@pytest.fixture(scope="module")
def env(runtime_config):
    return create_brax_h1_env(
        env_config=runtime_config.env,
        reward_config=runtime_config.reward,
        dr_config=runtime_config.domain_randomization,
        termination_config=runtime_config.termination,
    )


@pytest.fixture(scope="module")
def step_fn(env):
    return jax.jit(env.step)


@pytest.fixture(scope="module")
def reset_state(env):
    state = env.reset(jax.random.PRNGKey(0))
    return env.set_command(state, jnp.zeros(3))


def action_for(env, offsets: dict[str, float], action_scale: float) -> jnp.ndarray:
    action = jnp.zeros(env.action_size)
    for joint, offset in offsets.items():
        joint_id = mujoco.mj_name2id(env._mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        actuator = int(np.where(env._mj_model.actuator_trnid[:, 0] == joint_id)[0][0])
        action = action.at[actuator].set(offset / action_scale)
    return action


def test_model_has_a_floor(env):
    assert mujoco.mj_name2id(env._mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0


def test_feet_start_on_the_floor(env, reset_state):
    assert np.array(env.foot_contact(reset_state.pipeline_state)).tolist() == [True, True]
    soles = reset_state.pipeline_state.site_xpos[env._foot_site_ids][:, 2]
    assert np.all(np.abs(np.array(soles)) < 0.01)


def test_upright_robot_is_not_terminated(env, reset_state):
    _, info = compute_termination(reset_state.pipeline_state.qpos, env._termination_config)
    assert abs(float(info["roll"])) < 1e-3
    assert abs(float(info["pitch"])) < 1e-3
    assert not bool(env._compute_termination(reset_state.pipeline_state))


@pytest.mark.parametrize("axis", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
@pytest.mark.parametrize("angle", [0.3, -0.3, 0.7, -0.7])
def test_termination_respects_tilt_limits(env, reset_state, axis, angle):
    quat = jnp.concatenate([jnp.array([jnp.cos(angle / 2)]), jnp.sin(angle / 2) * jnp.array(axis)])
    qpos = reset_state.pipeline_state.qpos.at[3:7].set(quat)
    expected = abs(angle) > 0.5
    assert bool(env._compute_termination(reset_state.pipeline_state.replace(qpos=qpos))) == expected


def test_robot_does_not_fall_through_floor_in_first_second(env, step_fn, reset_state, runtime_config):
    state = reset_state
    for _ in range(50):
        state = step_fn(state, jnp.zeros(env.action_size))
        assert float(state.done) == 0.0
    height = float(state.pipeline_state.qpos[2])
    assert height > runtime_config.termination.min_height
    assert abs(height - env.diagnostics_summary()["initial_height"]) < 0.1


def test_lifting_left_leg_breaks_left_contact_only(env, step_fn, reset_state, runtime_config):
    lift = action_for(env, {"left_hip_pitch": -0.6, "left_knee": 1.2}, runtime_config.env.action_scale)
    state = reset_state
    for _ in range(10):
        state = step_fn(state, lift)
    assert np.array(env.foot_contact(state.pipeline_state)).tolist() == [False, True]
    assert float(state.info["air_time"][0]) > 0.0


def test_observation_layout(env, reset_state):
    assert env.obs_version == OBS_VERSION == 2
    assert reset_state.obs.shape == (env.observation_size,)
    assert env.observation_size == observation_size(19, 19) == 73
    # At reset, joint positions are expressed relative to the default pose.
    assert np.allclose(np.array(reset_state.obs[:19]), 0.0, atol=1e-5)


def test_gait_clock_advances(env, step_fn, reset_state, runtime_config):
    state = step_fn(reset_state, jnp.zeros(env.action_size))
    expected = (reset_state.info["phase"] + env.control_dt * gait_frequency(reset_state.info["command"], runtime_config.env)) % 1.0
    assert np.isclose(float(state.info["phase"]), float(expected))


def test_reset_and_step_share_state_structure(env, step_fn, reset_state):
    """Brax's scan/auto-reset needs identical pytrees from reset and step."""
    stepped = step_fn(reset_state, jnp.zeros(env.action_size))
    assert jax.tree.structure(stepped.metrics) == jax.tree.structure(reset_state.metrics)
    assert jax.tree.structure(stepped.info) == jax.tree.structure(reset_state.info)


def test_auto_reset_restarts_episode_bookkeeping(runtime_config):
    """Brax's AutoResetWrapper restores only pipeline_state and obs. Per-episode
    info must restart too, or step_count keeps growing across episodes."""
    from brax.envs.wrappers import training

    env_config = dataclasses.replace(runtime_config.env, episode_length=15)
    env = create_brax_h1_env(env_config, runtime_config.reward, None, runtime_config.termination)
    wrapped = training.wrap(env, episode_length=15, action_repeat=1)
    state = wrapped.reset(jax.random.split(jax.random.PRNGKey(0), 2))
    step = jax.jit(wrapped.step)
    dones = 0
    for _ in range(50):
        state = step(state, jnp.zeros((2, env.action_size)))
        dones += int(state.done.sum())
        assert int(state.info["step_count"].max()) < 15
    # Time limit of 15 steps over 50 steps: ~3 resets per env, not one per step.
    assert dones <= 2 * 4


def test_slower_commands_step_slower(runtime_config):
    slow = float(gait_frequency(jnp.array([0.25, 0.0, 0.0]), runtime_config.env))
    fast = float(gait_frequency(jnp.array([1.0, 0.0, 0.0]), runtime_config.env))
    assert slow < fast


def test_feet_start_at_the_h1_neutral_width(env, reset_state):
    from physics_ai.envs.h1_shared import feet_lateral_gap

    d = reset_state.pipeline_state
    gap = float(feet_lateral_gap(d.qpos, d.site_xpos[env._foot_site_ids]))
    # H1 legs hang 0.406 m apart at zero hip roll (hip pitch joints sit outboard).
    assert 0.38 < gap < 0.43


def test_reward_target_height_defaults_to_standing_height(env):
    assert env._reward_config.target_height == env.diagnostics_summary()["initial_height"]


def test_enabled_domain_randomization_is_refused():
    with pytest.raises(NotImplementedError):
        create_brax_h1_env(dr_config=DomainRandomizationConfig(friction_enabled=True))


def test_domain_randomization_defaults_off(runtime_config):
    assert not any(v for k, v in vars(runtime_config.domain_randomization).items() if k.endswith("_enabled"))
