"""Newton asset capability implementation."""

from typing import Any

from robofinals.core.backends.contract import AssetCompiler, BackendKind, RobotAssetVariants


class NewtonAssetCompiler:
    def __init__(self) -> None:
        pass

    def compile_articulation(self, spec: Any) -> Any:
        return _select_variant(spec, BackendKind.NEWTON)

    def compile_rigid_body(self, spec: Any) -> Any:
        return _select_variant(spec, BackendKind.NEWTON)

    def configure_robot(self, spec: Any, env_cfg: Any) -> None:
        selected = spec.select(BackendKind.NEWTON)
        if selected.configure_env is not None:
            selected.configure_env(env_cfg)

    def compile_robot(self, spec: Any) -> Any:
        return self.compile_articulation(spec)

    def compile_object(self, spec: Any) -> Any:
        return self.compile_rigid_body(spec)

    def rigid_object_cfg_kwargs(self) -> dict[str, Any]:
        return {}

    def bind_zero_dof_articulation(self) -> bool:
        return False

    def bind_fixed_joint_rigid_body(self) -> bool:
        return False

    def supports_deformable_solver(self, solver_cfg: Any) -> bool:
        return solver_cfg is not None

    def configure_deformable_cfg(self, cfg: Any, *, omniphysics: bool = False) -> None:
        if omniphysics:
            raise RuntimeError("OmniPhysics deformables require the PhysX backend")

    def instantiate_deformable(self, cfg: Any, *, omniphysics: bool = False) -> Any:
        self.configure_deformable_cfg(cfg, omniphysics=omniphysics)
        from isaaclab.assets import DeformableObject

        return DeformableObject(cfg)

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
        from isaaclab.assets import DeformableObjectCfg
        from .deformable_assets import (
            SchemaAwareDeformableReferenceCfg,
            SchemaAwareDeformableUsdFileCfg,
        )

        if usd_path is not None:
            spawn = SchemaAwareDeformableUsdFileCfg(
                usd_path=usd_path,
                scale=scale,
                schema=schema,
                tetmesh_source=tetmesh_source,
            )
            return DeformableObjectCfg(prim_path=prim_path, spawn=spawn)
        return DeformableObjectCfg(
            prim_path=prim_path,
            spawn=SchemaAwareDeformableReferenceCfg(
                schema=schema, tetmesh_source=tetmesh_source
            ),
            init_state=DeformableObjectCfg.InitialStateCfg(
                pos=initial_pose.position_xyz, rot=initial_pose.rotation_wxyz
            ),
        )


def _select_variant(spec: Any, backend: BackendKind) -> Any:
    if isinstance(spec, RobotAssetVariants):
        return spec.select(backend).articulation
    if hasattr(spec, "select"):
        return spec.select(backend)
    if isinstance(spec, dict) and "physx_path" in spec and "newton_path" in spec:
        return spec["newton_path"]
    return spec

__all__ = ["AssetCompiler", "NewtonAssetCompiler"]
