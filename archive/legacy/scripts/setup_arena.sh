#!/usr/bin/env bash
# setup_arena.sh — Initialize IsaacLab-Arena + VLA Arena extension
# Usage: bash scripts/setup_arena.sh
set -euo pipefail

ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/rocket/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64}"
ISAAC_PYTHON="${ISAAC_SIM_ROOT}/python.sh"
ARENA_ROOT="${ARENA_ROOT:-/home/rocket/Projects/IsaacLab-Arena}"
VLA_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== VLA Arena Setup ==="
echo "Isaac Sim:  ${ISAAC_SIM_ROOT}"
echo "Arena:      ${ARENA_ROOT}"
echo "VLA:        ${VLA_ROOT}"
echo ""

# ─── 1. Verify Isaac Sim installation ────────────────────────────────────────
if [ ! -f "${ISAAC_PYTHON}" ]; then
    echo "ERROR: Isaac Sim python.sh not found at ${ISAAC_PYTHON}"
    echo "Set ISAAC_SIM_ROOT to your Isaac Sim installation directory."
    exit 1
fi
echo "[1/7] Isaac Sim found at ${ISAAC_SIM_ROOT}"

# ─── 2. Initialize IsaacLab-Arena submodules ─────────────────────────────────
echo "[2/7] Initializing IsaacLab-Arena submodules..."
cd "${ARENA_ROOT}"
git submodule update --init --recursive

# ─── 3. Install IsaacLab into Isaac Sim's Python ─────────────────────────────
echo "[3/7] Installing IsaacLab..."
ISAACLAB_DIR="${ARENA_ROOT}/submodules/IsaacLab"
if [ -d "${ISAACLAB_DIR}" ]; then
    ln -sfn "${ISAAC_SIM_ROOT}" "${ISAACLAB_DIR}/_isaac_sim"
    export ISAACLAB_PATH="${ISAACLAB_DIR}"

    for DIR in "${ISAACLAB_DIR}"/source/isaaclab*/; do
        if [ -d "$DIR" ]; then
            echo "  Installing $(basename "$DIR")..."
            "${ISAAC_PYTHON}" -m pip install --no-deps -e "$DIR" 2>&1 | tail -1
        fi
    done

    "${ISAACLAB_DIR}/isaaclab.sh" -i
else
    echo "WARNING: IsaacLab submodule not found at ${ISAACLAB_DIR}"
    echo "Skipping IsaacLab installation."
fi

# ─── 4. Install IsaacLab core deps ───────────────────────────────────────────
echo "[4/7] Installing IsaacLab core dependencies..."
"${ISAAC_PYTHON}" -m pip install \
    warp-lang \
    flatdict \
    toml \
    "prettytable==3.3.0" \
    "gymnasium==1.2.1" \
    onnxruntime \
    lightwheel-sdk

# Install PINK IK library (pin-pink, not 'pink' which is a code formatter)
"${ISAAC_PYTHON}" -m pip install pin-pink "numpy<2"

# Remove conflicting pink.py if present (Isaac Sim ships a different 'pink' module)
PINK_PY="${ISAAC_SIM_ROOT}/kit/python/lib/python3.11/site-packages/pink.py"
if [ -f "${PINK_PY}" ]; then
    echo "  Removing conflicting pink.py..."
    rm -f "${PINK_PY}"
    rm -f "${PINK_PY}c" "${ISAAC_SIM_ROOT}/kit/python/lib/python3.11/site-packages/__pycache__/pink.cpython-311.pyc" 2>/dev/null || true
fi

# ─── 5. Install IsaacLab-Arena ────────────────────────────────────────────────
echo "[5/7] Installing IsaacLab-Arena..."
"${ISAAC_PYTHON}" -m pip install -e "${ARENA_ROOT}/"

# ─── 6. Install VLA Arena extension ──────────────────────────────────────────
echo "[6/7] Installing VLA Arena extension (isaaclab_arena_vla)..."
"${ISAAC_PYTHON}" -m pip install -e "${VLA_ROOT}/"

# ─── 7. Install VLA + LeRobot dependencies ───────────────────────────────────
echo "[7/7] Installing VLA + LeRobot dependencies..."
"${ISAAC_PYTHON}" -m pip install \
    "transformers>=4.40,<4.50" \
    "timm>=0.9.10,<1.0" \
    bitsandbytes \
    accelerate \
    lerobot \
    scipy

echo ""
echo "=== Setup complete ==="
echo ""
echo "Smoke test:"
echo "  ISAACLAB_PATH=${ISAACLAB_DIR} ${ISAAC_PYTHON} scripts/test_arena_env.py --headless --num_envs 1 --num_steps 20"
echo ""
echo "Run with Arena's policy_runner:"
echo "  ISAACLAB_PATH=${ISAACLAB_DIR} ${ISAAC_PYTHON} ${ARENA_ROOT}/isaaclab_arena/examples/policy_runner.py \\"
echo "    --environment isaaclab_arena_vla.environments.so100_pick_and_place:SO100PickAndPlaceEnvironment \\"
echo "    --headless so100_pick_and_place"
