# Experiment 2: Physics & Reward Debugging

## Initial Problem
Training showed no learning. Reward flatlined at 0.138.
```
Update     30 | Reward:   0.138 | KL:  0.9304
Update    396 | Reward:   0.138 (no change)
```

Humanoid appeared frozen. No movement learning despite stable KL.

## Root Cause 1: Default Pose Not Applied

**Finding:** Humanoid spawned with all joints at zero (straight legs). Falls immediately.

**Code Issue:**
```python
def _get_default_qpos(self) -> jnp.ndarray:
    return jnp.array(self.mj_model.qpos0)  # XML defaults, ignores config
```

Config had proper standing pose but was never applied:
```yaml
left_hip_pitch: -0.4
left_knee: 0.8
left_ankle: -0.4
```

**Fix:** Apply config defaults in `_get_default_qpos()`. Humanoid now spawns in bent-knee standing pose.

**Result:** No improvement. Humanoid still didn't move.

## Root Cause 2: Action Scale Too Small

**Finding:** Actuators are torque motors with ±200 Nm range.
```xml
<motor name="left_hip_yaw" ctrlrange="-200 200"/>
```

Policy outputs [-1, 1] scaled by 0.5 = **±0.5 Nm torques**.

Humanoid needs 50-100 Nm to move joints. Humanoid physically incapable of motion.

**Fix:** `action_scale: 0.5 → 50.0`

**Result:** Still no improvement. Physics correct but no learning signal.

## Root Cause 3: Reward Too Sparse

**Finding:** Velocity tracking reward was exponentially harsh.
```python
exp(-vel_error / 0.25)  # target=1.0, current=0 → reward=0.018
```

Humanoid got almost zero gradient when far from target velocity.

Breakdown of 0.138 reward:
- Alive bonus: 1.0 × 0.1 scaling = 0.10
- Upright/height: ~0.03
- Velocity tracking: ~0.008 (negligible)

**Fix:**
```yaml
reward_scaling: 0.1 → 1.0        # 10x larger gradients
velocity exp_scale: 0.25 → 2.0   # More forgiving
velocity weight: 1.0 → 2.0       # Emphasize movement
```

Also reduced command difficulty:
```yaml
vx_range: [-1.0, 1.0] → [0.0, 0.5]    # Forward only
vy_range: [-0.5, 0.5] → [-0.2, 0.2]   # Smaller lateral
vyaw_range: [-1.0, 1.0] → [-0.3, 0.3] # Smaller turning
```

**Result:** Training worked. Reward jumped to 3.2 (23x improvement).

## Final Training Stats
```
Best reward: 3.2198
Training time: 5.36 hours
Average FPS: 674
Early stopped after 100 updates without improvement
```

Reward components achieved:
- Alive: 1.0
- Velocity tracking: ~1.3 / 2.0 (65%)
- Upright/height: ~0.3 / 0.3 (100%)
- Yaw/other: ~0.6 / 1.3 (46%)

## Conclusion

Humanoid learned to stand and partially track velocity. Plateaued at 3.2 reward.

**What worked:**
- Proper standing pose initialization
- Correct action scaling (50 Nm)
- Forgiving reward structure
- Easier command curriculum

**What didn't:**
- Initial sparse rewards (exp scale too small)
- Difficult velocity targets for initial learning

**Remaining issues:**
- Velocity tracking incomplete (only 65% of target)
- Policy became deterministic (entropy → 6.6)
- Early stopping triggered too soon

## Next Steps

Disable early stopping. Increase exploration:
```yaml
early_stopping: true → false
entropy_coef: 0.005 → 0.02
learning_rate: 5e-5 → 3e-4
num_epochs: 2 → 4
patience: 100 → 200
```

Re-enable domain randomization after stable baseline achieved.
