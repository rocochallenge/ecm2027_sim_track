#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/runtime_common.sh"
mkdir -p recordings
exec "$PYTHON_BIN" -m robofinals.applications.teleop.entrypoint --task_config=galbot --viz=kit --physics_backend=physx "$@"
