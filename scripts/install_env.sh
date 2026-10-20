#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ISAACLAB_REVISION=28a37cecdd433c22d9eabd6a5954add9f13a8951
ISAACLAB_ROOT="$PROJECT_ROOT/.deps/IsaacLab"
CONSTRAINTS="$PROJECT_ROOT/requirements/runtime-constraints.txt"
COMPAT_PATCH="$PROJECT_ROOT/patches/isaaclab-6.1.11-isaacsim-6.0.1.patch"

if [[ "${1:-}" == --help ]]; then
  printf 'Usage: bash scripts/install_env.sh\nActivate a dedicated Python 3.12 Conda/virtual environment first.\nDownloads Isaac Sim, CUDA PyTorch wheels, and public Isaac Lab source.\nDoes not install system packages, accept licenses, or change drivers.\n'
  exit 0
fi
if [[ $# -gt 0 ]]; then
  printf 'Unexpected arguments. Use --help for usage.\n' >&2
  exit 2
fi

"$PYTHON_BIN" -c 'import os, platform, sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"; assert platform.system() == "Linux" and platform.machine() == "x86_64", "Linux x86_64 is required"; assert os.environ.get("CONDA_PREFIX") or sys.prefix != sys.base_prefix, "Activate a dedicated Conda/virtual environment first"'
command -v git >/dev/null
git lfs version
nvidia-smi
cd "$PROJECT_ROOT"
git lfs pull
git lfs fsck

if [[ ! -d "$ISAACLAB_ROOT" ]]; then
  mkdir -p "$(dirname "$ISAACLAB_ROOT")"
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch v3.0.0-beta2 https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_ROOT"
fi
if [[ "$(git -C "$ISAACLAB_ROOT" rev-parse HEAD)" != "$ISAACLAB_REVISION" ]]; then
  printf 'Unexpected Isaac Lab revision at %s; refusing to overwrite it.\n' "$ISAACLAB_ROOT" >&2
  exit 1
fi
if [[ -z "$(git -C "$ISAACLAB_ROOT" status --porcelain --untracked-files=no)" ]]; then
  git -C "$ISAACLAB_ROOT" apply --check "$COMPAT_PATCH"
  git -C "$ISAACLAB_ROOT" apply "$COMPAT_PATCH"
elif ! git -C "$ISAACLAB_ROOT" diff --no-ext-diff --binary | cmp -s - "$COMPAT_PATCH"; then
  printf 'Isaac Lab changes differ from the documented compatibility patch; refusing to overwrite them.\n' >&2
  exit 1
fi

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install -c "$CONSTRAINTS" torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
"$PYTHON_BIN" -m pip install -c "$CONSTRAINTS" 'isaacsim[all,extscache]==6.0.1' --extra-index-url https://pypi.nvidia.com
"$PYTHON_BIN" -m pip install toml
"$PYTHON_BIN" -m pip install --no-build-isolation -c "$CONSTRAINTS" --extra-index-url https://pypi.nvidia.com \
  -e "$ISAACLAB_ROOT/source/isaaclab" \
  -e "$ISAACLAB_ROOT/source/isaaclab_physx" \
  -e "$ISAACLAB_ROOT/source/isaaclab_ov" \
  -e "$ISAACLAB_ROOT/source/isaaclab_visualizers" \
  -e "$ISAACLAB_ROOT/source/isaaclab_assets" \
  -e "$ISAACLAB_ROOT/source/isaaclab_tasks" \
  -e "$PROJECT_ROOT"
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/check_environment.py"
printf '\nInstallation checks passed. Run the documented Chamber teleop/replay/evaluation runtime checks next.\n'
