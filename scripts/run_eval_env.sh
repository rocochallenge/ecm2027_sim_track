#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/runtime_common.sh"
exec "$PYTHON_BIN" -m robofinals.applications.eval.entrypoints.env_server --remote_protocol=ipc --ipc_host=127.0.0.1 --ipc_port=50000 --enable_cameras "$@"
