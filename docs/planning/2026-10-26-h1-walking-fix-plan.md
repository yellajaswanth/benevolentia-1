# H1 Walking Fix Plan

## Why the project failed so far

> **Update (implementing Phase 1):** two bigger bugs turned up, both present since the first commit:
> - **No floor.** The env loaded `h1.xml`, which has no ground plane, so the robot was in free fall in every run. It now loads `scene.xml`.
> - **Every episode ended on step 1.** The termination check computed roll = π for an upright robot (`arctan2(g_y, g_z)` with gravity (0, 0, −1)), so `|roll| > 0.5` was always true.
>
> Also fixed: the configured 0.88 m reset height started the feet 10 cm under the floor (it is now derived from the pose, about 0.98 m), and the reward rotated an already body-frame angular velocity a second time.
>
> No previous training result says anything about whether the reward or PPO settings worked.
>
> **Update (first GPU runs, 2026-10-27):** three more bugs turned up during training:
> - Brax `batch_size` defaulted to `num_envs × rollout_length`, so one PPO update cost ~134M env steps. It is now `num_envs / num_minibatches`.
> - Brax's auto-reset restores only physics and the observation, so `step_count` carried over between episodes. After ~8M total steps every step counted as the time limit, and run 1 collapsed. The env now resets its per-episode info itself.
> - `scripts/evaluate.py` rebuilt the policy without observation normalisation, so every evaluation and video showed a broken policy.
>
> The run 1 swing-height penalty was too weak to produce stepping, so it was replaced by a Playground-style `feet_phase` reward, trained on curriculum stage A (fixed 0.5 m/s). Run 2 (200M steps, 32 min on one RTX 4090) walks: full 20 s episodes, 0.48–0.49 m/s, alternating 1.25 Hz gait.
>
> **Update (run 3: stage B + natural-gait changes):** forward commands 0–1 m/s plus 10% stand-still; gait clock scaled with speed (0.75 + 0.5·v Hz); straighter default pose with arms hanging; 6 cm swing target; mild height and bob penalties; new torso, arm-swing and stance-width terms. 200M steps in 34 min. No falls at stand, 0.25, 0.5 and 1.0 m/s; arms swing 16–23° with the opposite leg, torso twist down from 14° to 4–6°, 20–30% double support. Remaining: 7–9% speed undershoot, 14–28° heading drift over 20 s (no heading term), choppy ~2 Hz stepping at 1 m/s and stiff knees (action range ±0.3 rad). Also fixed: fixed-command evaluations were being re-sampled every 10 s.

1. **The reward pays for standing still.** With `velocity_tracking.exp_scale: 2.0` and commands of 0 to 0.5 m/s, a robot standing perfectly still earns about 95% of the velocity reward. On top of that it gets `alive: 1.0` every step and a −10 penalty for falling. "Don't fall" is the best strategy available.
2. **Nothing teaches it to step.** The reward has no foot terms (air time, contact timing, swing height, slip). The observation has no gait clock and no previous action, and the foot geoms have no names, so foot contact can't be measured at all.
3. **Training got far too few steps.** The only full run reached about 13M steps at about 670 steps/s. Working humanoid walking policies typically need 200M or more on a GPU.
4. **Hidden problems in the setup.** The Brax wrapper accepts `dr_config` but never applies it. The config loader turns domain randomisation **on** whenever a key is missing. Brax uses its tiny default network, (32,)×4. Two environments still exist side by side (`h1_env.py` and `brax_wrapper.py`).

Each phase below has a pass/fail check. Don't start a phase until the one before it passes.

---

## Phase 0: Prove the GPU pipeline on a known-good H1 task (1–2 days)

Goal: separate "our environment is wrong" from "our GPU/Brax/MJX setup is wrong".

- On RunPod, install `mujoco_playground` and train its H1 joystick locomotion task with its own default PPO config.
- Record wall-clock time, steps/s, the reward curve and a video.
- Keep its reward and observation code open as a reference for Phases 1–2.

**Pass:** the reference policy walks forward in the video. Throughput is well over 50k steps/s.
**If it fails:** fix the RunPod/JAX/MJX setup before going further. It's not a reward problem.

## Phase 1: Environment fixes (2–3 days)

Files: `assets/unitree_h1/h1.xml`, `physics_ai/envs/brax_wrapper.py`, `physics_ai/envs/h1_shared.py`, `physics_ai/config.py`.

1. **Make feet observable.**
   - Name the foot geoms (`left_foot1..3`, `right_foot1..3`).
   - Add `left_foot` and `right_foot` sites at the sole.
   - Add a helper `foot_contact(data) -> (2,) bool`, from MJX contact distance on the foot geoms.
   - Add a helper `foot_height(data) -> (2,)`, from the site z-position.
2. **Gait clock.**
   - Store `phase` in `state.info` and advance it by `dt_ctrl * gait_freq` every step. Start with `gait_freq ≈ 1.25 Hz`.
   - Left and right legs run half a cycle apart.
3. **Observation v2.** Current observation plus:
   - sin/cos of each foot's phase;
   - previous action;
   - base linear velocity in the robot's own frame. This is privileged information, which is fine for a sim-only baseline.

   Version the observation layout so old checkpoints fail loudly instead of silently misbehaving.
4. **One environment.** Make the Brax wrapper the only one. Delete `h1_env.py`, `scripts/train.py` and the custom PPO in `physics_ai/agents/`, or move them to `legacy/`.
5. **Domain randomisation: honest defaults.**
   - Change the loader so DR defaults to `False` when a key is missing.
   - Either wire `dr_config` into `reset`/`step` (deferred to Phase 5) or make the wrapper raise an error if any DR flag is on.

**Pass:** `scripts/diagnostics.py` reports foot contact correctly:
- both feet down at reset;
- one foot up when the swing leg's hip and knee are commanded by hand.

## Phase 2: Reward rewrite (2–3 days)

File: `physics_ai/rewards/walking.py`. Weights follow `unitree_rl_gym`'s H1 and the Playground task; treat them as starting points.

| Term | Form | Weight |
|---|---|---|
| Track forward/sideways velocity | `exp(-‖v_xy − cmd_xy‖² / 0.25)` | +1.0 |
| Track yaw rate | `exp(-(ω_z − cmd_ω)² / 0.25)` | +0.5 |
| Contact matches gait clock | +1 per foot whose contact matches its phase (stance vs swing) | +0.2 |
| Feet air time | reward air time on touchdown, only when commanded to move | +1.0 |
| Swing foot height | `−(h_foot − 0.08)²` for swing feet | −20 |
| Feet slip | `‖v_foot_xy‖²` for feet in contact | −0.05 |
| Orientation | `‖gravity_xy‖²` | −1.0 |
| Base height | `(z − 0.98·target)²` | −10 |
| Vertical velocity | `v_z²` | −2.0 |
| Roll/pitch rate | `‖ω_xy‖²` | −0.05 |
| Action rate | `‖a − a_prev‖²` | −0.01 |
| Torques | `‖τ‖²` | −1e-5 |
| Hip yaw/roll deviation | `‖q_hip_yaw,roll − default‖²` | −1.0 |
| Alive | constant | +0.15 |
| Termination | on fall | −10 (as now) |

- Remove the `reward_scaling` knob from the env so it stays at 1.0. Rescaling the reward hid the real numbers in Experiment 2.
- Log every term separately in Brax metrics. This already happens for a few terms; extend it to all of them.

**Pass: a reward-sanity unit test** (`tests/test_reward_sanity.py`). This test encodes the lesson of this project:
- With command `vx = 0.5`, a robot standing still must score **≤ 40%** of the velocity reward that perfect tracking gets.
- The total reward for standing still must be lower than for a synthetic walking state with alternating contacts and the correct speed.

## Phase 3: Training config (1 day)

File: `configs/h1_walking.yaml`, `scripts/train_brax.py`.

- `num_timesteps`: 200M to start; allow up to 500M.
- `num_envs`: 8192.
- `unroll_length`: 20. `discounting`: 0.97. `entropy_cost`: 0.01. Learning rate 3e-4.
- `num_minibatches`: 32. `num_updates_per_batch`: 4.
- Pass `network_factory` with policy layers (512, 256, 128) and value layers (512, 256, 128).
- Keep `episode_length` at 1000 (20 s at 50 Hz).
- **GPU guard:** `train_brax.py` exits if `jax.default_backend() != "gpu"`, unless `--allow-cpu` is passed, which should only be used for smoke tests.
- Save a checkpoint and a short evaluation video every N evals, so a run can be judged before it finishes.

## Phase 4: Curriculum (1–2 weeks of runs)

Each stage starts from the previous stage's checkpoint.

| Stage | Commands | Pass |
|---|---|---|
| A. Fixed forward | `vx = 0.5`, `vy = 0`, `yaw = 0`, no resampling | Mean `vx` within ±20% of command; survives the full 20 s in >95% of episodes; feet visibly alternate |
| B. Forward range + standing | `vx ∈ [0, 1.0]`, 10% of episodes commanded zero | Tracks each 0.25 m/s bin within ±20%; stands still on zero command |
| C. Turning | add `yaw ∈ [−0.5, 0.5]` | yaw-rate error < 0.15 rad/s |
| D. Sideways | add `vy ∈ [−0.3, 0.3]` | lateral error < 0.1 m/s |

**Reward-hacking alarm**, checked in `scripts/evaluate.py`: flag any run where the mean speed is below 0.1 m/s while commanded 0.5, whatever the reward says.

## Phase 5: Robustness (after Stage D)

- Wire domain randomisation into the wrapper: friction, link masses, motor strength, push forces, action latency. Add them one at a time, comparing against a frozen clean baseline.
- Add observation noise.
- Pass: Stage A–D criteria still met with full randomisation; recovers from a 50 N push.

## Cleanup (any time)

- Archive the scripted ZMP attempts: `scripts/run_programmatic.py` and the untracked `programmatic_walking*.mp4` and `zmp_walking.mp4` files.
- Update the README so it describes only the single Brax path.

## Timeline

| Phase | Effort |
|---|---|
| 0 | 1–2 days |
| 1–2 | ~1 week |
| 3 | 1 day |
| 4 | 1–2 weeks (mostly GPU time) |
| 5 | 1 week |
