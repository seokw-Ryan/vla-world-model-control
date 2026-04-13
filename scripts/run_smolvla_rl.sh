#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "${ISAACSIM:-}" ]]; then
  echo "ISAACSIM is not set. Example:"
  echo "  export ISAACSIM=~/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64"
  exit 1
fi

cd "$PROJECT_ROOT"
"$ISAACSIM/python.sh" scripts/train_smolvla_rl.py "$@"
