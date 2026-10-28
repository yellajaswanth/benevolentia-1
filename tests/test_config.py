from physics_ai.config import load_runtime_config


def test_each_ppo_update_uses_one_unroll_per_env():
    """batch_size * num_minibatches must equal num_envs; the old default
    (num_envs * rollout_length) made one update cost ~134M env steps."""
    ppo = load_runtime_config("configs/h1_walking.yaml").brax_ppo
    assert ppo["batch_size"] * ppo["num_minibatches"] == ppo["num_envs"]
    steps_per_update = ppo["num_envs"] * ppo["unroll_length"]
    assert ppo["num_timesteps"] // steps_per_update >= 1000
