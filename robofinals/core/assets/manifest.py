from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from robofinals.core.backends.common import BackendKind


@dataclass(frozen=True)
class AssetManifest:
    asset_id: str
    backend_variants: Mapping[str, str] = field(default_factory=dict)
    articulation_root: str | None = None
    joints: tuple[str, ...] = ()
    links: tuple[str, ...] = ()
    sensors: tuple[str, ...] = ()
    required_extensions: tuple[str, ...] = ()

    def path_for(self, backend: BackendKind | str) -> str | None:
        key = BackendKind.parse(backend).value
        return self.backend_variants.get(key) or self.backend_variants.get("default")


class AssetValidationError(ValueError):
    """Raised when an asset manifest cannot satisfy a backend contract."""


@dataclass(frozen=True)
class AssetValidationReport:
    asset_id: str
    backend: str
    path: str | None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors


class AssetValidator:
    def __init__(self, *, exists: Callable[[str], bool] | None = None) -> None:
        self.exists = exists or (lambda path: Path(path).exists())

    def validate_paths(self, manifest: AssetManifest, backend: BackendKind | str) -> AssetValidationReport:
        kind = BackendKind.parse(backend)
        path = manifest.path_for(kind)
        errors: list[str] = []
        if not path:
            errors.append(f"asset {manifest.asset_id!r} has no {kind.value} variant")
        elif not self.exists(path):
            errors.append(f"asset path does not exist: {path}")
        return AssetValidationReport(manifest.asset_id, kind.value, path, tuple(errors))

    def validate_joints(self, manifest: AssetManifest, available: set[str]) -> AssetValidationReport:
        missing = tuple(joint for joint in manifest.joints if joint not in available)
        return AssetValidationReport(
            manifest.asset_id,
            "unknown",
            None,
            tuple(f"missing joint: {joint}" for joint in missing),
        )

    def validate_backend(self, manifest: AssetManifest, backend: BackendKind | str) -> AssetValidationReport:
        return self.validate_paths(manifest, backend)

    def validate(self, manifest: AssetManifest, backend: BackendKind | str, *, available_joints: set[str] | None = None) -> AssetValidationReport:
        path_report = self.validate_paths(manifest, backend)
        if available_joints is None:
            return path_report
        joint_report = self.validate_joints(manifest, available_joints)
        return AssetValidationReport(
            path_report.asset_id,
            path_report.backend,
            path_report.path,
            path_report.errors + joint_report.errors,
            path_report.warnings,
        )

    def assert_valid(self, manifest: AssetManifest, backend: BackendKind | str, *, available_joints: set[str] | None = None) -> AssetValidationReport:
        report = self.validate(manifest, backend, available_joints=available_joints)
        if not report.valid:
            raise AssetValidationError("; ".join(report.errors))
        return report
