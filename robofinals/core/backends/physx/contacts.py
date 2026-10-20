"""PhysX contact capability implementation."""

from typing import Any

from robofinals.core.backends.contract import ContactProvider
from robofinals.core.backends import contact_common


class PhysXContactProvider(contact_common.ContactSensorAccess):
    def has_contact(self, source: str, target: str, *, threshold: float = 1e-8) -> Any:
        sensor = _sensor_name(self.env, getattr(source, "name", source), getattr(target, "name", target))
        return self.contact_detected(sensor, threshold=threshold)

    def _finalize_status(self, sensor_name, status):
        if not status.ready:
            return status
        sensor = (
            self.env.scene.sensors.get(sensor_name)
            if isinstance(sensor_name, str)
            else sensor_name
        )
        if hasattr(sensor, "contact_view") and sensor.contact_view is None:
            from dataclasses import replace

            return replace(status, ready=False, reason="backend_view_invalid")
        return status

    def check_contact(self, geoms_1, geoms_2):
        import re
        import torch
        from robofinals.utils import object_utils as ou
        sensor_path = _contact_sensor_name(geoms_1)
        sensors = self.env.scene.sensors
        if sensor_path not in sensors:
            return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        if self.env.common_step_counter == 1:
            if isinstance(geoms_2, str):
                target_paths = geoms_2
            else:
                target_paths = []
                for prim in ou.usd.get_prim_by_name(self.env.scene.stage.GetPseudoRoot(), geoms_2.name):
                    target_paths.append([str(cp.GetPrimPath()) for cp in ou.usd.get_prim_by_types(prim, ["Mesh", "Cube", "Cylinder"])])
            source_paths = _contact_sensor_paths_by_env(self.env.scene.sensors[sensor_path], self.env.num_envs)
            for env_id in range(self.env.num_envs):
                filters = [re.sub(r"env_\d+", f"env_{env_id}", target_paths)] if isinstance(geoms_2, str) else target_paths[env_id]
                self.env.cfg.isaaclab_arena_env.task.contact_queues[env_id].add(self.env.sim.physics_sim_view.create_rigid_contact_view(source_paths[env_id], filter_patterns=filters, max_contact_data_count=200))
        elif self.env.common_step_counter:
            views = [self.env.cfg.isaaclab_arena_env.task.contact_queues[i].pop() for i in range(self.env.num_envs)]
            return torch.tensor([_rigid_contact_view_has_contact(v, self.env.physics_dt) for v in views], device=self.env.device)
        return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)

    def receptacle_contact(self, obj_name, receptacle_name):
        import torch
        obj_sensor = self.env.scene.sensors[f"{obj_name}_contact"]
        rec_sensor = self.env.scene.sensors[f"{receptacle_name}_contact"]
        obj_paths = _contact_sensor_paths_by_env(obj_sensor, self.env.num_envs)
        rec_paths = _contact_sensor_paths_by_env(rec_sensor, self.env.num_envs)
        if self.env.common_step_counter > 1:
            views = [self.env.cfg.isaaclab_arena_env.task.contact_queues[i].pop() for i in range(self.env.num_envs)]
            return torch.tensor([_rigid_contact_view_has_contact(v, self.env.physics_dt) for v in views], device=self.env.device)
        for i in range(self.env.scene.num_envs):
            self.env.cfg.isaaclab_arena_env.task.contact_queues[i].add(self.env.sim.physics_sim_view.create_rigid_contact_view(obj_paths[i], [rec_paths[i]], max_contact_data_count=200))
        return torch.zeros(self.env.scene.num_envs, dtype=torch.bool, device=self.env.device)

    def uses_dynamic_contact_views(self) -> bool:
        return True


def _sensor_name(env: Any, source: str, target: str) -> str:
    sensors = getattr(getattr(env, "scene", None), "sensors", {})
    return target if target in sensors else f"{source}_contact"


def _contact_sensor_name(geoms_1) -> str:
    if isinstance(geoms_1, str):
        return f"{geoms_1}_contact"
    if hasattr(geoms_1, "task_name"):
        return f"{geoms_1.task_name}_contact"
    return f"{geoms_1.name}_contact"


__all__ = ["ContactProvider", "PhysXContactProvider"]


def _make_env_path_group(paths):
    return paths[0] if len(paths) == 1 else "(" + "|".join(paths) + ")"

def _contact_sensor_paths_by_env(sensor, num_envs):
    for view_name in ("contact_physx_view", "contact_view"):
        view = getattr(sensor, view_name, None)
        paths = getattr(view, "sensor_paths", None)
        if paths is not None:
            return list(paths)
    body_view = getattr(sensor, "body_physx_view", None) or getattr(sensor, "_body_physx_view", None)
    prim_paths = list(getattr(body_view, "prim_paths", []) or [])
    if not prim_paths:
        raise AttributeError("Could not resolve contact sensor paths")
    if len(prim_paths) == num_envs:
        return prim_paths
    num_sensors = int(getattr(sensor, "num_sensors", 0) or 0)
    if num_sensors <= 0 and num_envs > 0 and len(prim_paths) % num_envs == 0:
        num_sensors = len(prim_paths) // num_envs
    if num_sensors <= 0 or len(prim_paths) != num_envs * num_sensors:
        raise RuntimeError(f"Unexpected contact sensor prim path layout: paths={len(prim_paths)}, num_envs={num_envs}")
    return [_make_env_path_group(prim_paths[i * num_sensors:(i + 1) * num_sensors]) for i in range(num_envs)]

def _rigid_contact_view_has_contact(view, dt):
    try:
        forces = view.get_contact_data(dt)[0]
    except Exception:
        return False
    if hasattr(forces, "numpy"):
        forces = forces.numpy()
    else:
        import torch
        if isinstance(forces, torch.Tensor):
            forces = forces.detach().cpu().numpy()
    import numpy as np
    return bool(np.asarray(forces, dtype=np.float64).ravel().size and np.any(np.abs(np.asarray(forces, dtype=np.float64)) > 1e-8))
