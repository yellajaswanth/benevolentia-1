"""Reward sanity checks.

Earlier runs used a forgiving velocity kernel that paid a robot standing still
about 95% of the velocity reward, so PPO learned to stand. These tests pin
down the property that matters: when told to walk, walking must pay clearly
more than standing.
"""
from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import pytest

from physics_ai.config import load_runtime_config
from physics_ai.rewards.walking import RewardConfig, RewardInputs, compute_reward, compute_reward_terms

NU = 19
WALK_COMMAND = jnp.array([0.5, 0.0, 0.0])


def make_inputs(**overrides) -> RewardInputs:
    """A nominal upright robot at standing height with both feet down."""
    values = dict(
        base_lin_vel=jnp.zeros(3),
        base_ang_vel=jnp.zeros(3),
        projected_gravity=jnp.array([0.0, 0.0, -1.0]),
        base_height=jnp.array(0.98),
        command=WALK_COMMAND,
        action=jnp.zeros(NU),
        prev_action=jnp.zeros(NU),
        torques=jnp.zeros(NU),
        hip_offset=jnp.zeros(4),
        torso_offset=jnp.array(0.0),
        arm_swing_error=jnp.zeros(2),
        arm_posture_offset=jnp.zeros(6),
        feet_lateral_gap=jnp.array(0.30),
        foot_contact=jnp.array([True, True]),
        desired_contact=jnp.array([True, False]),
        first_contact=jnp.array([False, False]),
        air_time=jnp.zeros(2),
        foot_height=jnp.zeros(2),
        # Mid-swing of the right foot: the gait clock wants it at swing height.
        foot_height_target=jnp.array([0.0, 0.06]),
        foot_vel_xy=jnp.zeros((2, 2)),
        done=jnp.array(False),
    )
    values.update(overrides)
    return RewardInputs(**values)


def standing_still() -> RewardInputs:
    return make_inputs()


def walking_on_command() -> RewardInputs:
    """Mid-swing of a correct gait: right foot up at swing height, left planted."""
    return make_inputs(
        base_lin_vel=jnp.array([0.5, 0.0, 0.0]),
        foot_contact=jnp.array([True, False]),
        foot_height=jnp.array([0.0, 0.06]),
    )


def walking_touchdown() -> RewardInputs:
    """The step the swing foot lands after 0.4 s in the air."""
    return make_inputs(
        base_lin_vel=jnp.array([0.5, 0.0, 0.0]),
        foot_contact=jnp.array([True, True]),
        desired_contact=jnp.array([True, True]),
        first_contact=jnp.array([False, True]),
        air_time=jnp.array([0.0, 0.4]),
        foot_height_target=jnp.zeros(2),
    )


@pytest.fixture(params=["defaults", "yaml"])
def config(request) -> RewardConfig:
    config = RewardConfig() if request.param == "defaults" else load_runtime_config("configs/h1_walking.yaml").reward
    # The env fills target_height from the derived standing height; stand in for it here.
    return dataclasses.replace(config, target_height=0.98)


def test_standing_earns_at_most_40_percent_of_velocity_reward(config):
    standing = compute_reward_terms(standing_still(), config)["tracking_lin_vel"]
    perfect = compute_reward_terms(walking_on_command(), config)["tracking_lin_vel"]
    assert float(standing) <= 0.4 * float(perfect)


def test_walking_beats_standing_when_commanded_to_walk(config):
    standing, _ = compute_reward(standing_still(), config)
    walking, _ = compute_reward(walking_on_command(), config)
    touchdown, _ = compute_reward(walking_touchdown(), config)
    assert float(walking) > float(standing) + 0.5
    assert float(touchdown) > float(standing) + 0.5


def test_standing_wins_when_commanded_to_stand(config):
    stand_cmd = jnp.zeros(3)
    standing, _ = compute_reward(make_inputs(command=stand_cmd), config)
    stepping_in_place, _ = compute_reward(
        make_inputs(
            command=stand_cmd,
            foot_contact=jnp.array([True, False]),
            foot_height=jnp.array([0.0, 0.08]),
        ),
        config,
    )
    assert float(standing) > float(stepping_in_place)


def test_feet_phase_rewards_lifting_the_swing_foot(config):
    standing = compute_reward_terms(standing_still(), config)["feet_phase"]
    stepping = compute_reward_terms(walking_on_command(), config)["feet_phase"]
    assert float(stepping) > float(standing) + 0.3
    assert float(standing) < 0.7


def test_wide_stance_is_penalized(config):
    hip_width, _ = compute_reward(walking_on_command(), config)
    waddle, _ = compute_reward(dataclasses.replace(walking_on_command(), feet_lateral_gap=jnp.array(0.45)), config)
    assert float(waddle) < float(hip_width)


def test_arms_following_opposite_leg_beat_frozen_arms(config):
    swinging = compute_reward_terms(make_inputs(arm_swing_error=jnp.zeros(2)), config)["arm_swing"]
    frozen = compute_reward_terms(make_inputs(arm_swing_error=jnp.array([0.2, -0.2])), config)["arm_swing"]
    assert config.arm_swing * float(swinging) > config.arm_swing * float(frozen)


def test_air_time_reward_is_never_negative(config):
    short_hop = make_inputs(first_contact=jnp.array([False, True]), air_time=jnp.array([0.0, 0.02]))
    assert float(compute_reward_terms(short_hop, config)["feet_air_time"]) >= 0.0


def test_air_time_only_rewarded_when_moving(config):
    hop = dict(first_contact=jnp.array([False, True]), air_time=jnp.array([0.0, 0.4]))
    assert float(compute_reward_terms(make_inputs(command=jnp.zeros(3), **hop), config)["feet_air_time"]) == 0.0
    assert float(compute_reward_terms(make_inputs(**hop), config)["feet_air_time"]) > 0.0


def test_falling_is_worse_than_any_single_step_of_standing(config):
    standing, _ = compute_reward(standing_still(), config)
    fallen, _ = compute_reward(make_inputs(done=jnp.array(True)), config)
    assert float(fallen) < float(standing) - 5.0


def test_yaml_keeps_strict_tracking_sigma():
    reward = load_runtime_config("configs/h1_walking.yaml").reward
    assert reward.tracking_sigma <= 0.25
