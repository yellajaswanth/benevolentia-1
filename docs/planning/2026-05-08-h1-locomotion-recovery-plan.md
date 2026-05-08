# H1 Locomotion Recovery Plan: 6-8 Week Brax PPO Baseline Reset

## Summary

This plan resets the project around one canonical goal: a trustworthy simulated walking baseline for Unitree H1 using **Brax PPO as the only active training stack**. The core principle is that training, evaluation, video generation, checkpointing, and metrics must all describe the **same environment, same physics assumptions, and same policy format**.

## Milestones

### Milestone 0: Canonical System Definition
- Canonical stack: `Brax PPO`
- Canonical environment: Brax wrapper backed by the same reset, observation, reward, and termination logic as the active locomotion environment
- Canonical config surface: all physically meaningful parameters loaded from config and printed at runtime
- Canonical evaluation: one checkpoint format, one evaluator, one video path

### Milestone 1: Physics and Environment Integrity
- Move standing pose, initial height, action scaling, reward target height, and termination thresholds into one config-driven path
- Remove drift between `UnitreeH1Env`, Brax wrapper, and evaluation/video reset logic
- Add deterministic diagnostics for reset parity, action semantics, termination behavior, observation shape, and actuator interpretation

### Milestone 2: Standing Baseline
- Train on posture stability first
- Use near-zero commands and track standing duration, fall rate, torso pitch/roll envelope, and control smoothness

**Hypothesis H1:** if environment and control semantics are coherent, standing should succeed quickly; failure here is more likely a physics/config problem than a learning-capacity problem.

### Milestone 3: Fixed-Speed Forward Walking
- Introduce a single forward target speed
- Disable lateral/yaw variation and command resampling
- Track achieved-vs-commanded forward velocity, reward breakdown, episode length, and distance traveled

**Hypothesis H2:** fixed-speed walking should succeed before multi-command tracking; failure here most likely indicates reward or action semantics, not insufficient policy capacity.

### Milestone 4: Bounded Command Tracking
- Expand to a limited command curriculum: forward range first, yaw second, lateral last
- Add evaluation sweeps over command bins and a milestone scorecard

**Hypothesis H3:** command diversity should come only after stable fixed-speed walking; broader commands earlier were likely part of prior non-learning.

### Milestone 5: Robustness Gate
- Reintroduce perturbations incrementally and keep a frozen clean-baseline benchmark
- Compare baseline, per-perturbation, and combined-perturbation runs

**Hypothesis H4:** the project historically overstated robustness because domain-randomization configuration existed without a single canonical path proving when and how it was active.

## Implementation Notes

- Canonical checkpoint format is `brax_policy.pkl`
- Legacy custom PPO is preserved only as reference and should not be used for Brax evaluation
- Runtime artifacts must include the resolved config and run metadata for every training run
- Diagnostics are required before claiming a training failure is an algorithm issue

## Assumptions

- This plan optimizes for a **simulation baseline**, not hardware deployment
- Brax PPO is the sole active training/evaluation stack
- Historical failure was primarily caused by pipeline inconsistency across env, config, training, eval, and docs
