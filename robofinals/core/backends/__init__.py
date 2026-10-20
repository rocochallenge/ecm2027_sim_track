"""Backend selection and isolated PhysX/Newton capability adapters.

Use ``make_asset_compiler(kind)`` with ``RobotAssetVariants`` for robot assets;
use ``make_backend_runtime(kind)`` only when the full runtime bundle is needed.
"""

from .common import BackendKind, resolve_backend_kind
from .contract import (
    AssetCompiler,
    ContactSensorRef,
    ArticulationVariants,
    BackendAssetVariants,
    BackendLaunchProfile,
    BackendRuntime,
    ContactProvider,
    RendererProvider,
    RenderCapabilities,
    RecordingPolicy,
    RigidBodyVariants,
    RobotAssetConfig,
    RobotAssetVariants,
    SimulationCompiler,
    StateSynchronizer,
)
from .factory import attach_backend_runtime, make_asset_compiler, make_backend_runtime

__all__ = [
    "AssetCompiler",
    "ContactSensorRef",
    "ArticulationVariants",
    "BackendAssetVariants",
    "BackendKind",
    "BackendLaunchProfile",
    "BackendRuntime",
    "ContactProvider",
    "RendererProvider",
    "RenderCapabilities",
    "RecordingPolicy",
    "RigidBodyVariants",
    "RobotAssetConfig",
    "RobotAssetVariants",
    "SimulationCompiler",
    "StateSynchronizer",
    "make_backend_runtime",
    "make_asset_compiler",
    "attach_backend_runtime",
    "resolve_backend_kind",
]
