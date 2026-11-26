#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/runtime_common.sh"
exec "$PYTHON_BIN" -m robofinals.applications.replay.entrypoints.replay_main_local \
  --replay_mode state --record --enable_cameras --headless --device cpu "$@"
