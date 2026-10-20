#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  printf 'Usage: %s DATASET.hdf5 [action|joint_target] [replay options...]
' "$0" >&2
  exit 2
fi
DATASET="$(realpath "$1")"
REPLAY_MODE="${2:-action}"
if [[ $# -ge 2 ]]; then shift 2; else shift; fi
source "$(dirname "${BASH_SOURCE[0]}")/runtime_common.sh"
mkdir -p output
exec "$PYTHON_BIN" -m robofinals.applications.replay.entrypoints.replay_action_demo --dataset_file="$DATASET" --num_envs=1 --replay_mode="$REPLAY_MODE" --headless --without_image --device=cpu --record "$@"
