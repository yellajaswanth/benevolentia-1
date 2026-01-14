## Preliminary results observations

Catastrophically high kl divergence values were observed in training logs
```
Update      0 | KL: 110232.8047
Update     10 | KL: 191611.9062
Update     20 | KL: 230684.9062
```
KL should be around: 0.01-0.1
Diagnosis: Policy is changing drastically between updates. This is violating PPO's trust region constraint. Training is unstable from start.

Reward also flatlined
```
Reward:   0.123 → 0.128 (only 0.005 improvement over 39M timesteps)
```
Policy never learned anything meaningful. 

Entropy is stuck
```
Entropy:  26.960 → 27.427 (essentially constant)
```
For the action dimension and the gaussian policy, this is bad. Incorrect initialization or learning issues.

## Resolution Proposal

[Brax](https://github.com/google/brax) has robust PPO training implementation.
Current setup has 982 FPS at 113 hours for 400M steps. Brax apparently can do 50k+ simulations under an hour.



