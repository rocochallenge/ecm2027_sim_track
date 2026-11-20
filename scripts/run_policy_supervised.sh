#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/runtime_common.sh"
exec "$PYTHON_BIN" -m robofinals.applications.eval.policy_supervisor "$@"
