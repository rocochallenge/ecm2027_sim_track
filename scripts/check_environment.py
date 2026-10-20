from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "vendor" / "isaaclab_arena"))
    sys.path.insert(0, str(project_root))
    failures = []

    def report(name: str, passed: bool, detail: str) -> None:
        print(f"{'OK' if passed else 'FAIL'} {name}: {detail}")
        if not passed:
            failures.append(name)

    report("Python", sys.version_info[:2] == (3, 12), sys.version.split()[0])
    report("platform", platform.system() == "Linux" and platform.machine() == "x86_64", platform.platform())
    for requirement in (project_root / "requirements" / "runtime-constraints.txt").read_text().splitlines():
        package_name, expected = requirement.split("==")
        try:
            actual = importlib.metadata.version(package_name)
            report(package_name, actual.split("+")[0] == expected, f"{actual}; expected {expected}")
        except importlib.metadata.PackageNotFoundError:
            report(package_name, False, "not installed")
    for module_name in ("isaaclab", "isaaclab_physx", "isaaclab_ov", "isaaclab_visualizers", "isaaclab_assets", "isaaclab_tasks", "robofinals", "robofinals_tasks", "isaaclab_arena"):
        spec = importlib.util.find_spec(module_name)
        report(module_name, spec is not None, str(spec.origin if spec else "not found"))
    try:
        distribution = importlib.metadata.distribution("isaaclab")
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        expected_url = (project_root / ".deps" / "IsaacLab" / "source" / "isaaclab").as_uri()
        report("Isaac Lab source", direct_url.get("url") == expected_url, direct_url.get("url", "not an editable source installation"))
        revision = subprocess.run(["git", "-C", str(project_root / ".deps" / "IsaacLab"), "rev-parse", "HEAD"], capture_output=True, text=True)
        report("Isaac Lab revision", revision.returncode == 0 and revision.stdout.strip() == "28a37cecdd433c22d9eabd6a5954add9f13a8951", revision.stdout.strip() or revision.stderr.strip())
        changes = subprocess.run(["git", "-C", str(project_root / ".deps" / "IsaacLab"), "diff", "--no-ext-diff", "--binary"], capture_output=True, text=True)
        expected_patch = (project_root / "patches" / "isaaclab-6.1.11-isaacsim-6.0.1.patch").read_text()
        report("Isaac Lab compatibility patch", changes.returncode == 0 and changes.stdout == expected_patch, "must match the two documented dependency metadata changes")
    except importlib.metadata.PackageNotFoundError:
        report("Isaac Lab source", False, "not installed")
    try:
        import torch

        report("CUDA", torch.cuda.is_available(), f"torch CUDA {torch.version.cuda}")
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda").add_(1)
            torch.cuda.synchronize()
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception as error:
        report("CUDA", False, str(error))
    asset_root = project_root / "robofinals" / "data" / "assets"
    for directory in ("Scene", "galbot_one_golf_description", "roco_chamber_service/Sponge027"):
        asset_files = [path for path in (asset_root / directory).rglob("*") if path.is_file()]
        pointers = []
        for path in asset_files:
            with path.open("rb") as asset_file:
                if asset_file.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    pointers.append(str(path.relative_to(project_root)))
        report(directory, bool(asset_files) and not pointers, f"{len(asset_files)} files; LFS pointers: {pointers}")
    print(f"DISPLAY: {os.environ.get('DISPLAY', 'unset (desktop teleop needs X11)')}")
    print("Preflight only: Kit startup, VR input, replay and policy rollout are separate checks.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
