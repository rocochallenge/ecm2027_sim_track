"""Newton contact capability implementation."""

from typing import Any

from robofinals.core.backends.contract import ContactProvider
from robofinals.core.backends import contact_common


class NewtonContactProvider(contact_common.ContactSensorAccess):
    def has_contact(self, source: str, target: str, *, threshold: float = 1e-8) -> Any:
        sensors = getattr(getattr(self.env, "scene", None), "sensors", {})
        source_name = getattr(source, "name", source)
        target_name = getattr(target, "name", target)
        sensor = target_name if target_name in sensors else f"{source_name}_contact"
        return self.contact_detected(sensor, threshold=threshold)

    def check_contact(self, geoms_1, geoms_2):
        return _check_contact_newton_sensor(self.env, geoms_1, geoms_2)

    def receptacle_contact(self, obj_name, receptacle_name):
        return self.contact_detected(f"{obj_name}_contact") & self.contact_detected(f"{receptacle_name}_contact")

    def uses_dynamic_contact_views(self) -> bool:
        return False

__all__ = ["ContactProvider", "NewtonContactProvider"]


def _contact_endpoint_name(endpoint):
    if isinstance(endpoint, str):
        return endpoint.removesuffix("_contact")
    return getattr(endpoint, "task_name", None) or getattr(endpoint, "name", None)

def _check_contact_newton_sensor(env, geoms_1, geoms_2, *, threshold=1e-8):
    source_name = _contact_endpoint_name(geoms_1)
    target_name = _contact_endpoint_name(geoms_2)
    task = getattr(getattr(getattr(env, "cfg", None), "isaaclab_arena_env", None), "task", None)
    declared_pairs = tuple(getattr(task, "pairwise_contacts", ()) or ())
    if source_name and target_name:
        from robofinals.utils.object_utils import check_declared_contact_sensor
        for pair in ((source_name, target_name), (target_name, source_name)):
            if pair in declared_pairs:
                return check_declared_contact_sensor(env, pair, threshold)
    sensor_name = f"{source_name}_contact" if source_name else None
    if sensor_name is None or sensor_name not in env.scene.sensors:
        import torch
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env.backend_runtime.contacts.contact_detected(sensor_name, threshold=threshold)
