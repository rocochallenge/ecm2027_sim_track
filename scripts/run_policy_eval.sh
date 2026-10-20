#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  printf 'Usage: %s POLICY_CONFIG.yml [evaluation options...]
' "$0" >&2
  exit 2
fi
POLICY_CONFIG="$(realpath "$1")"
shift
source "$(dirname "${BASH_SOURCE[0]}")/runtime_common.sh"
exec "$PYTHON_BIN" -m robofinals.applications.eval.entrypoints.policy --config="$POLICY_CONFIG" "$@"
