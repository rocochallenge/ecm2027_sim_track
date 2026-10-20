"""Opt-in application of IsaacLab/Newton compatibility patches."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any


class CompatibilityError(RuntimeError):
    """Raised when the requested patch set cannot be safely enabled."""


@dataclass(frozen=True)
class CompatibilityReport:
    enabled: bool
    isaaclab_version: str | None
    isaacsim_version: str | None
    patches: tuple[str, ...]


_PATCHES = (
    "patch_warp_to_torch_accept_torch_tensor",
    "patch_camera_env_mask_has_any",
    "patch_reset",
    "patch_reset_to",
    "patch_configclass",
    "patch_recorder_manager_ep_meta",
    "patch_recorder_manager_joint_targets",
    "patch_step",
    "patch_yaml_load",
    "patch_reward_manager",
    "patch_create_teleop_device",
    "patch_isaaclab_tasks_mdp",
)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


class CompatibilityLayer:
    """Enable the legacy patch set exactly once after Kit startup."""

    _active = False
    _report: CompatibilityReport | None = None

    def __init__(
        self,
        *,
        backend: str | None = None,
        required_isaaclab: str | None = None,
        required_isaacsim: str | None = None,
    ) -> None:
        from robofinals.core.backends.common import resolve_backend_kind

        self.backend = resolve_backend_kind(backend)
        self.required_isaaclab = required_isaaclab
        self.required_isaacsim = required_isaacsim

    def inspect(self) -> CompatibilityReport:
        return CompatibilityReport(
            enabled=type(self)._active,
            isaaclab_version=_package_version("isaaclab"),
            isaacsim_version=_package_version("isaacsim"),
            patches=tuple(_PATCHES),
        )

    def enable(self) -> CompatibilityReport:
        if type(self)._active:
            return type(self)._report or self.inspect()
        report = self.inspect()
        self._check_version(report.isaaclab_version, self.required_isaaclab, "isaaclab")
        self._check_version(report.isaacsim_version, self.required_isaacsim, "isaacsim")
        try:
            from robofinals.utils import monkey_patch

            monkey_patch.apply_all_patches()
            from robofinals.core.backends import make_backend_runtime

            make_backend_runtime(self.backend).simulation.install_compat_patches()
        except Exception as exc:
            raise CompatibilityError(
                "Unable to enable IsaacLab compatibility patches. "
                "Call this only after AppLauncher has started Kit."
            ) from exc
        type(self)._active = True
        type(self)._report = CompatibilityReport(
            enabled=True,
            isaaclab_version=report.isaaclab_version,
            isaacsim_version=report.isaacsim_version,
            patches=tuple(_PATCHES),
        )
        return type(self)._report

    @staticmethod
    def _check_version(actual: str | None, expected: str | None, package: str) -> None:
        if expected is not None and actual is None:
            raise CompatibilityError(f"{package} is not installed; cannot enable compatibility layer")
        if expected is not None and actual != expected:
            raise CompatibilityError(f"{package} version {actual!r} does not match required {expected!r}")
