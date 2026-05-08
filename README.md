# PhysicsAI: Unitree H1 Locomotion Baseline

GPU-accelerated locomotion training stack for the Unitree H1 humanoid using MuJoCo MJX with Brax PPO as the canonical training and evaluation path.

## Features

- **MuJoCo MJX**: Hardware-accelerated physics simulation with XLA support
- **Brax PPO**: Canonical training backend for the active locomotion baseline
- **Legacy Custom PPO**: Preserved for reference only, not part of the recommended path
- **Domain Randomization**: Comprehensive sim-to-real robustness (friction, mass, motor strength, push forces, latency)
- **LocoMuJoCo Compatible**: Standardized interface for benchmarking

## Quick Start

### Prerequisites

- Python 3.10, 3.11, or 3.12 (3.13 not yet supported)
- **Apple Silicon Mac**: Requires ARM-native Python (see setup below)
- **Linux with NVIDIA GPU**: CUDA 12+ recommended for best performance

---

## Installation

### Option A: Apple Silicon Mac (M1/M2/M3/M4)

Apple Silicon requires ARM-native Python. Using x86 Python via Rosetta will cause JAX errors.

#### Step 1: Install Miniforge (ARM-native Conda)

```bash
# Download ARM-native Miniforge
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
chmod +x Miniforge3-MacOSX-arm64.sh
./Miniforge3-MacOSX-arm64.sh -b -p ~/miniforge3

# Initialize conda for your shell
~/miniforge3/bin/conda init zsh  # or: ~/miniforge3/bin/conda init bash
source ~/.zshrc  # or: source ~/.bashrc

# Verify ARM architecture
python -c "import platform; print(platform.machine())"  # Should print: arm64
```

#### Step 2: Create Environment

```bash
# Create and activate environment
conda create -n physics_ai python=3.12 -y
conda activate physics_ai

# Verify architecture again
python -c "import platform; print(platform.machine())"  # Must be: arm64
```

#### Step 3: Install JAX for Apple Silicon

```bash
# Install JAX (CPU version for Mac)
pip install jax jaxlib

# Optional: Metal GPU acceleration (experimental)
pip install jax-metal
```

#### Step 4: Install Package

```bash
cd physics_ai
pip install -e .
```

#### Troubleshooting ARM Issues

If you see this error:
```
RuntimeError: This version of jaxlib was built using AVX instructions...
```

Your Python is x86, not ARM. Fix it:

```bash
# Check your Python architecture
file $(which python)
# Should show: "Mach-O 64-bit executable arm64"
# NOT: "Mach-O 64-bit executable x86_64"

# If x86, reinstall with Miniforge (Step 1 above)
```

---

### Option B: Linux with NVIDIA GPU (Recommended for Training)

```bash
cd physics_ai

# Preferred: deterministic Runpod/NVIDIA install with pinned CUDA 12 dependencies
bash scripts/setup_runpod.sh

# If you need to select a specific interpreter explicitly:
PYTHON_BIN=python3.11 bash scripts/setup_runpod.sh
```

This path uses [requirements-runpod-cu12.txt](/Users/jaswanthyella/Benevolentia-1/physics_ai/requirements-runpod-cu12.txt) to install the exact Brax + MJX + JAX CUDA stack first, then installs the repo in editable mode without re-resolving dependencies.

Manual equivalent:

```bash
cd physics_ai
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-runpod-cu12.txt
python -m pip install --no-deps -e .
```

---

### Option C: Linux/Intel Mac (CPU only)

```bash
cd physics_ai
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Training

### Download Robot Assets

```bash
python scripts/download_assets.py
```

### Train with Brax PPO (Recommended)

Brax PPO provides significantly faster and more stable training:

```bash
python scripts/train_brax.py --config configs/h1_walking.yaml --checkpoint-dir checkpoints
```

To run in background
```bash
nohup python -u scripts/train_brax.py --config configs/h1_walking.yaml --checkpoint-dir checkpoints --num-envs 512 > training.log 2>&1 &
```

**Expected Performance:**
- FPS: 50,000 - 500,000 (vs ~1,000 with custom PPO)
- Training time: 15-45 minutes for 400M steps
- Stable KL divergence: 0.01 - 0.05

#### Command-line Options

```bash
python scripts/train_brax.py \
    --config configs/h1_walking.yaml \
    --checkpoint-dir checkpoints \
    --seed 42 \
    --num-envs 8192
```

### Run Diagnostics First

```bash
python scripts/diagnostics.py --config configs/h1_walking.yaml
```

This verifies reset parity, actuator semantics, termination logic, observation shape, and domain-randomization sampling before you spend time on training.

### Custom PPO (Legacy Reference Only)

```bash
python scripts/train.py --config configs/h1_walking.yaml
```

Note: The custom PPO path is quarantined from the canonical Brax baseline. Do not use it to evaluate or compare Brax checkpoints.

---

## Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/brax_policy.pkl
```

Legacy custom-PPO evaluation has been moved behind `scripts/evaluate_legacy.py` and is intentionally not part of the recommended workflow.

---

## Project Structure

```
physics_ai/
├── configs/
│   └── h1_walking.yaml         # Training hyperparameters
├── physics_ai/
│   ├── envs/
│   │   ├── h1_env.py           # Core MJX environment
│   │   ├── brax_wrapper.py     # Brax-compatible wrapper
│   │   ├── h1_shared.py        # Shared H1 reset/obs/termination semantics
│   │   ├── wrappers.py         # Gymnasium wrappers
│   │   └── domain_rand.py      # Domain randomization
│   ├── agents/
│   │   ├── ppo.py              # Custom PPO agent
│   │   ├── networks.py         # Actor-Critic networks
│   │   └── rollout_buffer.py   # Experience buffer + GAE
│   ├── rewards/
│   │   └── walking.py          # Walking reward function
│   └── utils/
│       └── jax_utils.py        # JAX utilities
├── scripts/
│   ├── train_brax.py           # Brax PPO training (recommended)
│   ├── train.py                # Custom PPO training (legacy)
│   ├── diagnostics.py          # Canonical environment diagnostics
│   ├── evaluate.py             # Canonical Brax policy evaluation
│   ├── evaluate_legacy.py      # Legacy custom-PPO warning shim
│   └── download_assets.py      # Asset downloader
├── assets/
│   └── unitree_h1/             # Robot MJCF files
└── checkpoints/                # Saved models
```

---

## Configuration

### Brax PPO Parameters (`configs/h1_walking.yaml`)

```yaml
brax_ppo:
  learning_rate: 3.0e-4
  entropy_cost: 0.001          # Critical for stable KL
  discounting: 0.99
  unroll_length: 32
  num_minibatches: 32
  num_updates_per_batch: 4
  normalize_observations: true
  clipping_epsilon: 0.2
  gae_lambda: 0.95
```

### Environment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_envs` | 8192 | Parallel environments |
| `episode_length` | 1000 | Steps per episode |
| `control_decimation` | 4 | Physics steps per control step |
| `robot.action_scale` | 0.25 | Position-target delta scale around the default stance |
| `robot.initial_height` | 0.88 | Reset height for the athletic stance |

### Canonical Artifacts

Training writes:
- `checkpoints/brax_policy.pkl`: canonical Brax policy bundle for evaluation
- `checkpoints/brax_final.pkl`: raw params snapshot
- `checkpoints/resolved_config.yaml`: resolved runtime config used for the run
- `checkpoints/run_metadata.json`: git commit, config path, and seed metadata

### Domain Randomization

| Category | Parameter | Range |
|----------|-----------|-------|
| Physics | Friction | [0.2, 1.0] |
| Mass | Body mass | ±10% |
| Actuator | Motor strength | ±15% |
| External | Push forces | 0-50N |
| Latency | Action delay | 0-20ms |

---

## Dependencies

- Python 3.10-3.12
- MuJoCo 3.1+
- MuJoCo MJX 3.1+
- Brax 0.14+
- JAX 0.4.30+
- Flax 0.10+

### GPU Requirements

- **NVIDIA**: CUDA 12+ with cuDNN
- **Apple Silicon**: Metal (experimental via jax-metal)

---

## Troubleshooting

### JAX AVX Error on Mac

```
RuntimeError: This version of jaxlib was built using AVX instructions...
```

**Solution**: Install ARM-native Python via Miniforge (see Installation Option A).

### Low FPS During Training

If FPS is under 10,000:
1. Use Brax PPO (`train_brax.py`) instead of custom PPO
2. Increase `num_envs` (8192 or higher)
3. Ensure GPU is being utilized: `nvidia-smi` or check JAX devices

### KL Divergence Explosion

If KL > 1.0 during training:
1. Reduce `learning_rate` to 1e-4
2. Ensure `entropy_cost` is set (0.001 recommended)
3. Use Brax PPO which handles this automatically

---

## License

MIT
