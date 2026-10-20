"""PhysX asset capability implementation."""

from typing import Any

from robofinals.core.backends.contract import AssetCompiler, BackendKind, RobotAssetVariants


class PhysXAssetCompiler:
    def __init__(self) -> None:
        pass

    def compile_articulation(self, spec: Any) -> Any:
        return _select_variant(spec, BackendKind.PHYSX)

    def compile_rigid_body(self, spec: Any) -> Any:
        return _select_variant(spec, BackendKind.PHYSX)

    def configure_robot(self, spec: Any, env_cfg: Any) -> None:
        selected = spec.select(BackendKind.PHYSX)
        if selected.configure_env is not None:
            selected.configure_env(env_cfg)

    def compile_robot(self, spec: Any) -> Any:
        return self.compile_articulation(spec)

    def compile_object(self, spec: Any) -> Any:
        return self.compile_rigid_body(spec)

    def rigid_object_cfg_kwargs(self) -> dict[str, Any]:
        return {}

    def bind_zero_dof_articulation(self) -> bool:
        return True

    def bind_fixed_joint_rigid_body(self) -> bool:
        return True

    def supports_deformable_solver(self, solver_cfg: Any) -> bool:
        return False

    def configure_deformable_cfg(self, cfg: Any, *, omniphysics: bool = False) -> None:
        if omniphysics:
            from .omniphysics_deformable import OmniPhysicsDeformableObject

            cfg.class_type = OmniPhysicsDeformableObject

    def instantiate_deformable(self, cfg: Any, *, omniphysics: bool = False) -> Any:
        self.configure_deformable_cfg(cfg, omniphysics=omniphysics)
        from isaaclab.assets import DeformableObject

        return cfg.class_type(cfg) if omniphysics else DeformableObject(cfg)

    def make_deformable_cfg(
        self,
        *,
        prim_path: str,
        usd_path: str | None = None,
        scale: Any = None,
        initial_pose: Any = None,
        tetmesh_source: str = "collision",
        schema: str = "auto",
    ) -> Any:
        del tetmesh_source, schema
        from isaaclab.assets import DeformableObjectCfg
        from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

        kwargs = {}
        if usd_path is not None:
            kwargs["spawn"] = UsdFileCfg(usd_path=usd_path, scale=scale)
        if initial_pose is not None:
            kwargs["init_state"] = DeformableObjectCfg.InitialStateCfg(
                pos=initial_pose.position_xyz, rot=initial_pose.rotation_wxyz
            )
        return DeformableObjectCfg(prim_path=prim_path, **kwargs)


def _select_variant(spec: Any, backend: BackendKind) -> Any:
    if isinstance(spec, RobotAssetVariants):
        return spec.select(backend).articulation
    if hasattr(spec, "select"):
        return spec.select(backend)
    if isinstance(spec, dict) and "physx_path" in spec and "newton_path" in spec:
        return spec["physx_path"]
    return spec

__all__ = ["AssetCompiler", "PhysXAssetCompiler"]
