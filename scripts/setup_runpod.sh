#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$ROOT_DIR/requirements-runpod-cu12.txt}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

pick_python() {
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    return
  fi

  for candidate in python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      return
    fi
  done

  echo "No supported Python interpreter found. Expected python3.10, python3.11, or python3.12." >&2
  exit 1
}

pick_python

cd "$ROOT_DIR"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REQUIREMENTS_FILE"
python -m pip install --no-deps -e .

python - <<'PY'
import importlib

modules = [
    "jax",
    "jaxlib",
    "brax",
    "mujoco",
    "mujoco.mjx",
    "flax",
    "optax",
]

for module_name in modules:
    importlib.import_module(module_name)

import jax

print("Runpod setup verification")
print(f"  jax version: {jax.__version__}")
print(f"  detected devices: {[str(device) for device in jax.devices()]}")
PY
