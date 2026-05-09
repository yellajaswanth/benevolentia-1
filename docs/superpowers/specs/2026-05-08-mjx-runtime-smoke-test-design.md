# MJX Runtime Smoke Test And Diagnostics Instrumentation Design

## Summary

This design isolates the current Runpod runtime problem by separating:

1. raw MJX GPU bring-up and first physics execution
2. environment-wrapper construction and reset behavior
3. higher-level diagnostics assertions

The goal is to determine whether the remaining failure is inside JAX/MJX compilation on the GPU, inside the H1 environment wrapper, or inside downstream diagnostics logic.

## Context

The Runpod bootstrap path is now reproducible enough to install the project and a CUDA-enabled JAX stack. The current blocker has shifted from dependency resolution to runtime behavior:

- the project installs on the Runpod box
- JAX has detected `cuda:0`
- the existing diagnostics path does not complete promptly
- the previous runtime produced a `ptxas`-related failure under an older JAX stack
- after the stricter JAX upgrade, diagnostics remained active and consumed large GPU memory, but did not emit enough progress detail to localize the stall

## Recommendation

Implement two complementary verification tools:

1. a minimal `scripts/mjx_smoke_test.py` that exercises only the smallest MJX path required to validate GPU runtime behavior
2. staged progress logging in `scripts/diagnostics.py` so the existing diagnostics path becomes observable and attributable

This is preferred over further dependency churn because it reduces uncertainty before changing versions again.

## Architecture

### Component 1: MJX Smoke Test

Add a new standalone script:

- path: `scripts/mjx_smoke_test.py`
- purpose: validate raw MuJoCo model loading and MJX forward execution with explicit timing at each stage

The script will:

1. parse the existing runtime config
2. locate the same H1 XML asset used by the active environment
3. import `jax`, `mujoco`, and `mujoco.mjx`
4. print device information
5. load the MuJoCo model from XML
6. convert the model to MJX
7. create MJX data
8. apply the canonical reset pose
9. run a single `mjx.forward`
10. optionally run one `mjx.step`
11. print elapsed time after each stage

### Component 2: Diagnostics Instrumentation

Update `scripts/diagnostics.py` so it no longer behaves like a black box during first compilation.

The script will emit named progress checkpoints before and after:

1. config load
2. Unitree environment creation
3. Brax environment creation
4. raw reset
5. command injection
6. fixed action step
7. low-height termination check
8. tilt termination check
9. summary serialization

Each checkpoint will include:

- wall-clock timestamp
- elapsed time since script start
- short phase label

## Data Flow

### MJX Smoke Test Flow

1. Read config via the shared runtime-config loader.
2. Resolve H1 asset path from the same canonical config path used by training/eval.
3. Load MuJoCo model.
4. Build canonical reset `qpos` from shared reset helpers.
5. Push model and data into MJX.
6. Apply canonical pose and run forward physics.
7. Emit a structured summary to stdout.

### Diagnostics Flow

1. Start timer and install phase logger.
2. Build Unitree env and Brax env using current canonical config.
3. Execute each existing diagnostics check with pre/post phase markers.
4. Emit either:
   - a successful JSON summary, or
   - the last completed phase before failure/hang

## Interfaces

### `scripts/mjx_smoke_test.py`

CLI arguments:

- `--config`
- `--run-step` to optionally execute one `mjx.step`
- `--output-json` for machine-readable output

Outputs:

- human-readable progress lines
- optional JSON payload with timings, device info, and success/failure status

### `scripts/diagnostics.py`

CLI additions:

- `--verbose-phases` defaulting to enabled
- `--output-json` remains supported if already present or should be added if missing

Outputs:

- phase logs to stdout
- final JSON summary if execution completes

## Error Handling

### Smoke Test

If a stage fails:

- print the stage label that failed
- print elapsed time until failure
- return non-zero exit code
- if `--output-json` is provided, write a failure record with the stage name and exception string

### Diagnostics

If a stage fails:

- print the last completed phase
- print the currently active phase
- return non-zero exit code
- preserve the existing exception traceback for debugging value

This design avoids swallowing runtime failures while still making them attributable.

## Testing Strategy

### Local validation

1. `python -m py_compile` on both scripts
2. smoke test import path validation on local machine
3. diagnostics script validation on local machine where feasible

### Runpod validation

1. run `bash scripts/setup_runpod.sh`
2. run `python scripts/mjx_smoke_test.py --config configs/h1_walking.yaml`
3. if smoke test passes, run `python scripts/diagnostics.py --config configs/h1_walking.yaml`
4. compare the last successful phase against prior behavior

### Success Criteria

- if smoke test fails before `mjx.forward`, the issue is below the environment layer
- if smoke test passes but diagnostics stalls during env reset, the issue is in wrapper/reset logic
- if both pass, the runtime stack is likely healthy enough for short evaluation runs

## Scope

In scope:

- one new smoke-test script
- phase/timing instrumentation in diagnostics
- minimal README update documenting the smoke test

Out of scope:

- broad environment refactors
- another dependency-matrix sweep
- training changes
- reward or control tuning

## Assumptions

- Assumption A1: the most valuable next step is localization, not optimization
- Assumption A2: the current canonical config path should remain the source of truth for assets and reset semantics
- Assumption A3: first-call MJX compilation on the GPU may be expensive, so timing visibility matters as much as pass/fail status
