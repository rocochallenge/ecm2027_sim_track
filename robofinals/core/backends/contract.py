"""Capability ports and runtime composition shared by PhysX and Newton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Protocol, TypeVar, runtime_checkable

from .common import BackendKind, resolve_backend_kind


AssetT = TypeVar("AssetT")

@dataclass(frozen=True)
class ContactSensorRef:
    """Typed reference to a contact sensor declared by a robot or scene."""
    name: str


@dataclass(frozen=True)
class BackendLaunchProfile:
    """Pure data required before Kit/AppLauncher starts."""

    backend: BackendKind
    kit_overrides: tuple[str, ...] = ()
    required_extensions: tuple[str, ...] = ()
    experience: str | None = None


@dataclass(frozen=True)
class BackendAssetVariants(Generic[AssetT]):
    """Explicit PhysX/Newton asset pair."""
    physx: AssetT
    newton: AssetT

    def select(self, kind: BackendKind | str) -> AssetT:
        return self.newton if BackendKind.parse(kind) is BackendKind.NEWTON else self.physx

@dataclass(frozen=True)
class ArticulationVariants(BackendAssetVariants[AssetT]):
    """Backend variants for an articulation."""

@dataclass(frozen=True)
class RigidBodyVariants(BackendAssetVariants[AssetT]):
    """Backend variants for a rigid body."""

@dataclass(frozen=True)
class RobotAssetConfig(Generic[AssetT]):
    articulation: AssetT
    configure_env: Callable[[Any], None] | None = None

@dataclass(frozen=True)
class RobotAssetVariants(Generic[AssetT]):
    """Backend-owned configuration for one logical robot."""
    physx: RobotAssetConfig[AssetT]
    newton: RobotAssetConfig[AssetT]

    def select(self, kind: BackendKind | str) -> RobotAssetConfig[AssetT]:
        return self.newton if BackendKind.parse(kind) is BackendKind.NEWTON else self.physx

@runtime_checkable
class SimulationCompiler(Protocol):
    def build_sim_config(self, spec: Any) -> Any: ...

    def configure_task_env(self, env_cfg: Any) -> Any: ...

    def configure_before_env_make(self, owner: Any, env_cfg: Any) -> Any: ...

    def install_compat_patches(self) -> None: ...

    def set_scene_params(self, sim_cfg: Any, **kwargs: float | int) -> None: ...

    def configure_deformable_solver(
        self, env_cfg: Any, *, enabled: bool, solver_cfg: dict[str, Any] | None
    ) -> None: ...


@runtime_checkable
class AssetCompiler(Protocol):
    def compile_articulation(self, spec: Any) -> Any: ...

    def compile_rigid_body(self, spec: Any) -> Any: ...

    def configure_robot(self, spec: RobotAssetVariants[Any], env_cfg: Any) -> None: ...

    def compile_robot(self, spec: Any) -> Any: ...

    def compile_object(self, spec: Any) -> Any: ...

    def rigid_object_cfg_kwargs(self) -> dict[str, Any]: ...

    def bind_zero_dof_articulation(self) -> bool: ...

    def bind_fixed_joint_rigid_body(self) -> bool: ...

    def supports_deformable_solver(self, solver_cfg: Any) -> bool: ...

    def configure_deformable_cfg(self, cfg: Any, *, omniphysics: bool = False) -> None: ...

    def instantiate_deformable(self, cfg: Any, *, omniphysics: bool = False) -> Any: ...

    def make_deformable_cfg(
        self,
        *,
        prim_path: str,
        usd_path: str | None = None,
        scale: Any = None,
        initial_pose: Any = None,
        tetmesh_source: str = "collision",
        schema: str = "auto",
    ) -> Any: ...


@runtime_checkable
class ContactProvider(Protocol):
    def has_contact(self, source: ContactSensorRef | str, target: ContactSensorRef | str, *, threshold: float = 1e-8) -> Any: ...

    def force_magnitude(self, sensor: ContactSensorRef | str, *, threshold: float = 1e-8) -> Any: ...

    def read_contact(self, sensor: ContactSensorRef | str, *, strict: bool = False) -> Any: ...

    def sensor_status(self, sensor: ContactSensorRef | str) -> Any: ...

    def validate_scene_sensors(self) -> list[Any]: ...

    def required_sensors_ready(
        self, sensors: Any, *, require_sim_playing: bool = True
    ) -> tuple[bool, list[Any]]: ...

    def assert_required_sensors_ready(self, sensors: Any) -> None: ...

    def contact_detected(
        self, sensor: ContactSensorRef | str, *, threshold: float = 1e-8, **kwargs: Any
    ) -> Any: ...

    def uses_dynamic_contact_views(self) -> bool: ...
    def check_contact(self, geoms_1: Any, geoms_2: Any) -> Any: ...

    def receptacle_contact(self, obj_name: str, receptacle_name: str) -> Any: ...


@runtime_checkable
class StateSynchronizer(Protocol):
    def reset(self, env: Any, env_ids: Any = None) -> None: ...

    def sync_after_state_reset(self, env: Any) -> None: ...

    def reset_sim(self, env: Any) -> None: ...

    def write_robot_startup_state(self, robot: Any, position: Any, rotation_xyzw: Any, env_ids: Any) -> None: ...


@runtime_checkable
class RendererProvider(Protocol):
    @property
    def capabilities(self) -> "RenderCapabilities": ...

    def render(self, env: Any) -> Any: ...

    def render_idle(self, env: Any) -> Any: ...

    def should_skip_step_render(self, env: Any) -> bool: ...

    def camera_name_candidates(self, side: str) -> tuple[str, ...]: ...


@runtime_checkable
class RecordingPolicy(Protocol):
    def prepare_value(self, value: Any) -> Any: ...

    def should_defer_observation(self, env: Any) -> bool: ...

    def storage_device(self, env: Any) -> Any: ...

    def should_defer_camera_capture(self, args: Any) -> bool: ...

    def supports_animation_recording(self) -> bool: ...

    def attach_animation_on_close(self, env: Any, *, delivery_dir: str | None = None) -> None: ...

    def finish_animation(self, env: Any, *, delivery_dir: str | None = None) -> str | None: ...


@dataclass(frozen=True)
class RenderCapabilities:
    allow_reset_render: bool = True
    allow_texture_wait: bool = True


@dataclass(frozen=True)
class BackendRuntime:
    """Composition of independent backend capabilities.

    This object is a dependency bundle, not an application session.  It can be
    constructed twice in one process for different backends without shared state.
    """

    kind: BackendKind
    launch: BackendLaunchProfile
    simulation: SimulationCompiler
    assets: AssetCompiler
    contacts: ContactProvider
    state: StateSynchronizer
    renderer: RendererProvider
    recording: RecordingPolicy
