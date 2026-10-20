# Copyright 2025 Lightwheel Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.math import matrix_from_quat, euler_xyz_from_quat

import robofinals.utils.math_utils.transform_utils.numpy_impl as T
from robofinals.utils.isaac_data_compat import sim_quat_raw_to_xyzw_np, sim_quat_raw_to_xyzw_torch, as_torch
from robofinals.core.models.fixtures.fixture import Fixture
from robofinals.utils.usd_utils import OpenUsd as usd


def _asset_data_to_torch(field: Any) -> torch.Tensor:
    """Isaac Lab buffers may be ``torch.Tensor`` or ``wp.array``."""
    return as_torch(field)


def body_com_pos_w_per_env_first_link(data: Any) -> torch.Tensor:
    """(num_envs, 3) world COM of first/only body per instance."""
    t = _asset_data_to_torch(data.body_com_pos_w)
    if t.ndim == 3:
        return t[:, 0, :]
    if t.ndim == 2:
        return t
    msg = f"body_com_pos_w: unexpected ndim={t.ndim} shape={tuple(t.shape)}"
    raise RuntimeError(msg)


def body_com_pos_w_mean_over_bodies(data: Any) -> torch.Tensor:
    """(num_envs, 3); single-body rigid objects are (num_envs, 3) already."""
    t = _asset_data_to_torch(data.body_com_pos_w)
    if t.ndim == 3:
        return torch.mean(t, dim=1)
    if t.ndim == 2:
        return t
    msg = f"body_com_pos_w: unexpected ndim={t.ndim} shape={tuple(t.shape)}"
    raise RuntimeError(msg)


def body_com_pos_w_for_gripper_distance(data: Any) -> torch.Tensor:
    """(num_envs, num_bodies, 3) for distance broadcast; expands 2D rigid buffers."""
    t = _asset_data_to_torch(data.body_com_pos_w)
    if t.ndim == 2:
        return t.unsqueeze(1)
    if t.ndim == 3:
        return t
    msg = f"body_com_pos_w: unexpected ndim={t.ndim} shape={tuple(t.shape)}"
    raise RuntimeError(msg)


def body_com_quat_w_mean_over_bodies(data: Any) -> torch.Tensor:
    t = _asset_data_to_torch(data.body_com_quat_w)
    if t.ndim == 3:
        return torch.mean(t, dim=1)
    if t.ndim == 2:
        return t
    msg = f"body_com_quat_w: unexpected ndim={t.ndim} shape={tuple(t.shape)}"
    raise RuntimeError(msg)


def body_com_quat_w_per_env_first_link(data: Any) -> torch.Tensor:
    """(num_envs, 4) world orientation of first/only body per instance."""
    t = _asset_data_to_torch(data.body_com_quat_w)
    if t.ndim == 3:
        return t[:, 0, :]
    if t.ndim == 2:
        return t
    msg = f"body_com_quat_w: unexpected ndim={t.ndim} shape={tuple(t.shape)}"
    raise RuntimeError(msg)


def body_com_vel_w_per_env_first_link(data: Any) -> torch.Tensor:
    t = _asset_data_to_torch(data.body_com_vel_w)
    if t.ndim == 3:
        return t[:, 0, :]
    if t.ndim == 2:
        return t
    msg = f"body_com_vel_w: unexpected ndim={t.ndim} shape={tuple(t.shape)}"
    raise RuntimeError(msg)


def body_com_vel_w_mean_over_bodies(data: Any) -> torch.Tensor:
    t = _asset_data_to_torch(data.body_com_vel_w)
    if t.ndim == 3:
        return torch.mean(t, dim=1)
    if t.ndim == 2:
        return t
    msg = f"body_com_vel_w: unexpected ndim={t.ndim} shape={tuple(t.shape)}"
    raise RuntimeError(msg)


def check_contact_sensor(
    env: ManagerBasedEnv,
    sensor_name: str,
    threshold: float = 1.0e-8,
) -> torch.Tensor:
    """Check a pre-filtered pairwise ContactSensor force matrix."""
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env.backend_runtime.contacts.contact_detected(sensor_name, threshold=threshold, require_force_matrix=True)


def _pairwise_contact_sensor_name(contact_pair: tuple[str, str]) -> str:
    """Build the internal scene key for a declared contact pair."""
    source_name, target_name = contact_pair
    return f"pairwise_contact__{source_name}__{target_name}"


def check_declared_contact_sensor(
    env: ManagerBasedEnv,
    contact_pair: tuple[str, str],
    threshold: float = 1.0e-8,
) -> torch.Tensor:
    """Check the sensor registered for a declared source-target pair."""
    return check_contact_sensor(env, _pairwise_contact_sensor_name(contact_pair), threshold)


def _declared_contact_endpoint_path(task, endpoint_name: str) -> str:
    """Resolve an object or fixture endpoint to its cloned rigid-body path."""
    custom_prim = getattr(task, "spongebody_contact_prim_path", None)
    if endpoint_name == "spongebody" and custom_prim:
        return custom_prim

    object_sensor = task.contact_sensors.get(f"{endpoint_name}_contact")
    if object_sensor is not None:
        return object_sensor.prim_path

    fixture = task.fixture_refs.get(endpoint_name)
    if fixture is None:
        raise KeyError(f"Unknown pairwise contact endpoint: {endpoint_name!r}")

    pending = [fixture.prim]
    while pending:
        prim = pending.pop(0)
        if usd.is_rigidbody(prim):
            break
        pending.extend(prim.GetAllChildren())
    else:
        raise RuntimeError(f"Pairwise contact endpoint {endpoint_name!r} has no rigid-body prim")

    source_path = prim.GetPath().pathString
    default_prim = prim.GetStage().GetDefaultPrim()
    default_path = default_prim.GetPath().pathString if default_prim else "/World"
    if source_path == default_path:
        suffix = ""
    elif source_path.startswith(default_path + "/"):
        suffix = source_path[len(default_path):]
    else:
        raise ValueError(f"Prim {source_path!r} is not below default prim {default_path!r}")
    return f"{{ENV_REGEX_NS}}/Scene{suffix}"


def register_declared_contact_sensors(task) -> None:
    """Register backend-neutral, pair-filtered sensors declared by a task."""
    for contact_pair in task.pairwise_contacts:
        source_name, target_name = contact_pair
        sensor_name = _pairwise_contact_sensor_name(contact_pair)
        if sensor_name in task.contact_sensors:
            raise ValueError(f"Duplicate pairwise contact declaration: {contact_pair!r}")
        task.add_contact_sensor_cfg(
            sensor_name,
            ContactSensorCfg(
                prim_path=_declared_contact_endpoint_path(task, source_name),
                update_period=0.0,
                history_length=1,
                debug_vis=False,
                force_threshold=0.0,
                filter_prim_paths_expr=[_declared_contact_endpoint_path(task, target_name)],
            ),
        )


def array_to_string(array):
    """
    Converts a numeric array into the string format in mujoco.

    Examples:
        [0, 1, 2] => "0 1 2"

    Args:
        array (n-array): Array to convert to a string

    Returns:
        str: String equivalent of @array
    """
    return " ".join(["{}".format(x) for x in array])


def _resolve_obj_inside_scene_key(task, obj_name: str) -> tuple[Any | None, str]:
    """Map a task object or fixture name to scene articulation/rigid-object key."""
    obj = task.objects.get(obj_name)
    if obj is not None:
        return obj, obj_name

    fixture_ref = task.fixture_refs.get(obj_name)
    if fixture_ref is not None:
        return None, fixture_ref.name

    if obj_name in task.fixtures:
        return None, obj_name

    return None, obj_name


def obj_inside_of(env: ManagerBasedEnv, obj_name: str, fixture_id: str, partial_check: bool = False, th=0.2) -> torch.Tensor:
    """
    whether an object (another mujoco object) is inside of fixture. applies for most fixtures
    """

    task = env.cfg.isaaclab_arena_env.task
    obj, scene_obj_name = _resolve_obj_inside_scene_key(task, obj_name)
    if obj is None and not partial_check:
        partial_check = True

    fixture = task.get_fixture(fixture_id)

    # step 1: calculate fxiture points
    fixtr_int_regions = fixture.get_int_sites(relative=False)
    check = []
    for i in range(env.cfg.scene.num_envs):
        if scene_obj_name not in env.scene.articulations and scene_obj_name not in env.scene.rigid_objects:
            check.append(False)
            continue
        inside_of = False
        for reset_region in fixtr_int_regions.values():
            inside_of = True
            fixtr_p0, fixtr_px, fixtr_py, fixtr_pz = [r + env.scene.env_origins[i].cpu().numpy() for r in reset_region]
            u = fixtr_px[i] - fixtr_p0[i]
            v = fixtr_py[i] - fixtr_p0[i]
            w = fixtr_pz[i] - fixtr_p0[i]

            # get the position and quaternion of object
            if scene_obj_name in env.scene.articulations:
                pos_t = _asset_data_to_torch(env.scene.articulations[scene_obj_name].data.body_com_pos_w)
                quat_t = _asset_data_to_torch(env.scene.articulations[scene_obj_name].data.body_com_quat_w)
                obj_pos = (pos_t[i, 0, :] if pos_t.ndim == 3 else pos_t[i, :]).cpu().numpy()
                obj_quat = sim_quat_raw_to_xyzw_np(
                    (quat_t[i, 0, :] if quat_t.ndim == 3 else quat_t[i, :]).cpu().numpy()
                )
            else:
                pos_t = _asset_data_to_torch(env.scene.rigid_objects[scene_obj_name].data.body_com_pos_w)
                quat_t = _asset_data_to_torch(env.scene.rigid_objects[scene_obj_name].data.body_com_quat_w)
                obj_pos = (pos_t[i, 0, :] if pos_t.ndim == 3 else pos_t[i, :]).cpu().numpy()
                obj_quat = sim_quat_raw_to_xyzw_np(
                    (quat_t[i, 0, :] if quat_t.ndim == 3 else quat_t[i, :]).cpu().numpy()
                )

            if partial_check:
                obj_points_to_check = [obj_pos]
                th_u = th_v = th_w = 0.0
            else:
                # calculate 8 boundary points of object
                obj_points_to_check = obj.get_bbox_points(trans=obj_pos, rot=obj_quat)
                # threshold to mitigate false negatives: even if the bounding box point is out of bounds,
                base_alpha = th
                min_th = 1e-4
                lu = np.linalg.norm(u) if np.linalg.norm(u) > 0 else 0.0
                lv = np.linalg.norm(v) if np.linalg.norm(v) > 0 else 0.0
                lw = np.linalg.norm(w) if np.linalg.norm(w) > 0 else 0.0

                th_u = max(min_th, lu * base_alpha)
                th_v = max(min_th, lv * base_alpha)
                th_w = max(min_th, lw * base_alpha)

            for obj_p in obj_points_to_check:
                check1 = (
                    np.dot(u, fixtr_p0[i]) - th_u <= np.dot(u, obj_p) <= np.dot(u, fixtr_px[i]) + th_u
                )
                check2 = (
                    np.dot(v, fixtr_p0[i]) - th_v <= np.dot(v, obj_p) <= np.dot(v, fixtr_py[i]) + th_v
                )
                check3 = (
                    np.dot(w, fixtr_p0[i]) - th_w <= np.dot(w, obj_p) <= np.dot(w, fixtr_pz[i]) + th_w
                )

                if not (check1 and check2 and check3):
                    inside_of = False
                    break

            if inside_of:
                check.append(True)
                break

        if not inside_of:
            check.append(False)

    return torch.tensor(check, dtype=torch.bool, device=env.device)


# used for cabinets, cabinet panels, counters, etc.
def set_geom_dimensions(sizes: Dict[str, List[float]], positions: Dict[str, List[float]], geoms: Dict[str, List[Any]], rotated: bool = False):
    """
    set the dimensions of geoms in a fixture

    Args:
        sizes (dict): dictionary of sizes for each side

        positions (dict): dictionary of positions for each side

        geoms (dict): dictionary of geoms for each side

        rotated (bool): whether the fixture is rotated. Fixture may be rotated to make texture appear uniform
                        due to mujoco texture conventions
    """
    if rotated:
        # rotation trick to make texture appear uniform
        # see .xml file
        for side in sizes.keys():
            # not the best detection method, TODO: check euler
            if "door" in side or "trim" in side:
                sizes[side] = [sizes[side][i] for i in [0, 2, 1]]

    # set sizes and positions of all geoms
    for side in positions.keys():
        for geom in geoms[side]:
            if geom is None:
                raise ValueError("Did not find geom:", side)
            geom.set("pos", array_to_string(positions[side]))
            geom.set("size", array_to_string(sizes[side]))


def get_rel_transform(fixture_A: Fixture, fixture_B: Fixture) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gets fixture_B's position and rotation relative to fixture_A's frame
    """
    A_trans = np.array(fixture_A.pos)
    B_trans = np.array(fixture_B.pos)

    A_rot = np.array([0, 0, fixture_A.rot])
    B_rot = np.array([0, 0, fixture_B.rot])

    A_mat = T.euler2mat(A_rot)
    B_mat = T.euler2mat(B_rot)

    T_WA = np.vstack((np.hstack((A_mat, A_trans[:, None])), [0, 0, 0, 1]))
    T_WB = np.vstack((np.hstack((B_mat, B_trans[:, None])), [0, 0, 0, 1]))

    T_AB = np.matmul(np.linalg.inv(T_WA), T_WB)

    return T_AB[:3, 3], T_AB[:3, :3]


def transform_global_to_local(global_x: float, global_y: float, rot: float) -> Tuple[float, float]:
    """
    Transforms a global movement vector from a global frame (with rotation `rot`)
    into local coordinates (assumed to be rotation 0).
    Args:
        global_x (float): Movement along local x-axis
        global_y (float): Movement along local y-axis
        rot (float): Rotation of the local frame (in radians)
    Returns:
        (float, float): Transformed (local_x, local_y)
    """
    cos_yaw = torch.cos(rot)
    sin_yaw = torch.sin(rot)

    local_x = cos_yaw * global_x - sin_yaw * global_y
    local_y = sin_yaw * global_x + cos_yaw * global_y

    return local_x, local_y


def compute_rel_transform(A_pos: np.ndarray, A_mat: np.ndarray, B_pos: np.ndarray, B_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gets B's position and rotation relative to A's frame
    """
    T_WA = np.vstack((np.hstack((A_mat, A_pos[:, None])), [0, 0, 0, 1]))
    T_WB = np.vstack((np.hstack((B_mat, B_pos[:, None])), [0, 0, 0, 1]))

    T_AB = np.matmul(np.linalg.inv(T_WA), T_WB)

    return T_AB[:3, 3], T_AB[:3, :3]


def get_fixture_to_point_rel_offset(fixture: Fixture, point: np.ndarray, rot: float = None) -> np.ndarray:
    """
    get offset relative to fixture's frame, given a global point
    """
    global_offset = point - fixture.pos
    if rot is None:
        rot = fixture.rot
    T_WF = T.euler2mat([0, 0, rot])
    rel_offset = np.matmul(np.linalg.inv(T_WF), global_offset)
    return rel_offset


def get_pos_after_rel_offset(fixture: Fixture, offset: np.ndarray) -> np.ndarray:
    """
    Get the global position after applying an offset relative to the center of the fixture.
    Supports offset of shape (3,) or (N, 3).
    """
    fixture_rot = np.array([0, 0, fixture.rot])
    fixture_mat = T.euler2mat(fixture_rot)

    offset = np.asarray(offset)
    if offset.ndim == 1:
        # (3,)
        return fixture.pos + np.dot(fixture_mat, offset)
    elif offset.ndim == 2 and offset.shape[1] == 3:
        # (N, 3)
        return fixture.pos + np.dot(offset, fixture_mat.T)
    else:
        raise ValueError("offset must have shape (3,) or (N, 3)")


def project_point_to_line(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    logic copied from here: https://stackoverflow.com/a/61342198
    """
    AP = P - A
    AB = B - A
    result = A + np.dot(AP, AB) / np.dot(AB, AB) * AB

    return result


def point_in_fixture(point: np.ndarray, fixture: Fixture, only_2d: bool = False) -> bool:
    """
    check if point is inside of the exterior bounding boxes of the fixture

    Args:
        point (np.array): point to check

        fixture (Fixture): fixture object

        only_2d (bool): whether to check only in 2D
    """
    try:
        if isinstance(point, torch.Tensor):
            point = point.detach().cpu().numpy()
    except Exception as e:
        print(f"Error converting point to numpy array: {e}")
        pass

    p0, px, py, pz = fixture.get_ext_sites(relative=False)
    th = 0.00
    u = px - p0
    v = py - p0
    w = pz - p0
    check1 = np.dot(u, p0) - th <= np.dot(u, point) <= np.dot(u, px) + th
    check2 = np.dot(v, p0) - th <= np.dot(v, point) <= np.dot(v, py) + th
    check3 = np.dot(w, p0) - th <= np.dot(w, point) <= np.dot(w, pz) + th

    if only_2d:
        return check1 and check2
    else:
        return check1 and check2 and check3


from robofinals.utils.place_utils.usd_object import USDObject


def obj_in_region(
    obj: USDObject | Fixture | Any,
    obj_pos: np.ndarray,
    obj_quat: np.ndarray,
    p0: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray = None,
) -> bool:
    """
    check if object is in the region defined by the points.
    Uses either the objects bounding box or the object's horizontal radius
    """
    if isinstance(obj, USDObject) or isinstance(obj, Fixture):
        obj_points = obj.get_bbox_points(trans=obj_pos, rot=obj_quat)
    else:
        radius = obj.horizontal_radius
        obj_points = obj_pos + np.array(
            [
                [radius, 0, 0],
                [-radius, 0, 0],
                [0, radius, 0],
                [0, -radius, 0],
            ]
        )

    u = px - p0
    v = py - p0
    w = pz - p0 if pz is not None else None

    for point in obj_points:
        check1 = np.dot(u, p0) <= np.dot(u, point) <= np.dot(u, px)
        check2 = np.dot(v, p0) <= np.dot(v, point) <= np.dot(v, py)

        if not check1 or not check2:
            return False

        if w is not None:
            check3 = np.dot(w, p0) <= np.dot(w, point) <= np.dot(w, pz)
            if not check3:
                return False

    return True


def fixture_pairwise_dist(f1: Fixture, f2: Fixture) -> float:
    """
    Gets the distance between two fixtures by finding the minimum distance between their exterior bounding box points
    """
    f1_points = f1.get_ext_sites(all_points=True, relative=False)
    f2_points = f2.get_ext_sites(all_points=True, relative=False)

    all_dists = [np.linalg.norm(p1 - p2) for p1 in f1_points for p2 in f2_points]
    return np.min(all_dists)


def objs_intersect(
    obj: USDObject | Fixture | Any,
    obj_pos: np.ndarray,
    obj_quat: np.ndarray,
    other_obj: USDObject | Fixture | Any,
    other_obj_pos: np.ndarray,
    other_obj_quat: np.ndarray,
    margin: float = 0.03,
) -> bool:
    """
    check if two objects intersect using Separating Axis Theorem (SAT)

    ``margin`` (meters) inflates the acceptance band on every separating axis:
    a gap is only treated as a real separation when it exceeds ``margin``. This
    absorbs the small edge slop of the bbox proxies (e.g. a fixture's reset-region
    box is slightly smaller than its physical footprint), so an object resting on
    the rim of a tray is still reported as in contact.
    """
    obj_points = obj.get_bbox_points(trans=obj_pos, rot=obj_quat)
    other_obj_points = other_obj.get_bbox_points(
        trans=other_obj_pos, rot=other_obj_quat
    )

    face_normals = [
        obj_points[1] - obj_points[0],
        obj_points[2] - obj_points[0],
        obj_points[3] - obj_points[0],
        other_obj_points[1] - other_obj_points[0],
        other_obj_points[2] - other_obj_points[0],
        other_obj_points[3] - other_obj_points[0],
    ]

    intersect = True

    # noramlize length of normals
    for normal in face_normals:
        normal = np.array(normal) / np.linalg.norm(normal)

        obj_projs = [np.dot(p, normal) for p in obj_points]
        other_obj_projs = [np.dot(p, normal) for p in other_obj_points]

        # see if gap detected (a separation smaller than ``margin`` is ignored)
        if np.min(other_obj_projs) - np.max(obj_projs) > margin or np.min(
            obj_projs
        ) - np.max(other_obj_projs) > margin:
            intersect = False
            break
    return intersect


def normalize_joint_value(raw: float, joint_min: float, joint_max: float) -> float:
    """
    normalize raw value to be between 0 and 1
    """
    return (raw - joint_min) / (joint_max - joint_min)


def check_obj_in_receptacle(env: ManagerBasedEnv, obj_name: str, receptacle_name: str, th: float = 0.25) -> torch.Tensor:
    """
    check if object is in receptacle object based on threshold
    """
    recep = env.cfg.isaaclab_arena_env.task.objects[receptacle_name]
    is_contact = env.backend_runtime.contacts.receptacle_contact(obj_name, receptacle_name)

    if obj_name in env.scene.articulations:
        obj_pos = body_com_pos_w_per_env_first_link(env.scene.articulations[obj_name].data)
    elif obj_name in env.scene.rigid_objects:
        obj_pos = body_com_pos_w_per_env_first_link(env.scene.rigid_objects[obj_name].data)
    if receptacle_name in env.scene.articulations:
        recep_pos = body_com_pos_w_per_env_first_link(env.scene.articulations[receptacle_name].data)
    elif receptacle_name in env.scene.rigid_objects:
        recep_pos = body_com_pos_w_per_env_first_link(env.scene.rigid_objects[receptacle_name].data)

    is_closed = torch.norm(obj_pos[:, :2] - recep_pos[:, :2], dim=-1) < th  # (env_num, )
    return is_contact & is_closed


def check_fxtr_upright(env, fixture_name, th=15):
    """
    Check if the fixture is upright based on its rotation.
    """
    fixture_quat_sim = _asset_data_to_torch(env.scene.articulations[fixture_name].data.body_com_quat_w)
    fixture_quat_xyzw = torch.tensor(
        sim_quat_raw_to_xyzw_np(fixture_quat_sim.cpu().numpy()), device=env.device
    )
    roll, pitch, yaw = euler_xyz_from_quat(fixture_quat_xyzw, degrees=True)
    is_upright = (torch.abs(roll) < th) & (torch.abs(pitch) < th)

    return is_upright


def check_obj_upright(env, obj_name, th=15):
    """
    Check if the object is upright based on its rotation.
    """
    obj_quat = sim_quat_raw_to_xyzw_torch(_asset_data_to_torch(env.scene[obj_name].data.root_quat_w))
    roll, pitch, y = euler_xyz_from_quat(obj_quat)
    is_upright = (torch.abs(roll) < th) & (torch.abs(pitch) < th)

    return is_upright


def check_obj_scrubbed(env, sponge_name, obj_name):
    """
    Determine if the sponge is scrubbing the object by checking contact and movement.
    """
    # Check if sponge is in contact with bowl
    in_contact = check_obj_in_receptacle(env, sponge_name, obj_name)
    sponge_pos = body_com_pos_w_per_env_first_link(env.scene.rigid_objects[sponge_name].data)
    prev_sponge_pos = getattr(env, "prev_sponge_pos", sponge_pos)

    movement_vector = sponge_pos - prev_sponge_pos

    in_contact = check_contact(env, env.cfg.isaaclab_arena_env.task.objects[sponge_name], env.cfg.isaaclab_arena_env.task.objects[obj_name])

    sponge_still_inside = check_obj_in_receptacle(env, sponge_name, obj_name)
    env.prev_sponge_pos = sponge_pos

    scrubbing = (
        in_contact & sponge_still_inside & (torch.norm(movement_vector) > 0.001)
    )

    return scrubbing


def _resolve_receptacle_fixture(env: ManagerBasedEnv, receptacle_name: str):
    """Return a receptacle task object or scene fixture, or None."""
    task = env.cfg.isaaclab_arena_env.task
    if receptacle_name in task.objects:
        return task.objects[receptacle_name]
    fixtures = getattr(task, "fixtures", {})
    fixture_refs = getattr(task, "fixture_refs", {})
    if receptacle_name in fixtures:
        return fixtures[receptacle_name]
    if receptacle_name in fixture_refs:
        return fixture_refs[receptacle_name]
    return task.get_fixture(receptacle_name)


def _scene_asset_pos_w(env: ManagerBasedEnv, asset_name: str):
    if asset_name in env.scene.articulations:
        return body_com_pos_w_per_env_first_link(env.scene.articulations[asset_name].data)
    if asset_name in env.scene.rigid_objects:
        return body_com_pos_w_per_env_first_link(env.scene.rigid_objects[asset_name].data)
    return None


def _receptacle_pos_w(env: ManagerBasedEnv, receptacle_name: str) -> torch.Tensor:
    """World position (num_envs, 3) for a receptacle object, rigid body, or static fixture."""
    pos = _scene_asset_pos_w(env, receptacle_name)
    if pos is not None:
        return pos

    receptacle = _resolve_receptacle_fixture(env, receptacle_name)
    if receptacle is not None and getattr(receptacle, "pos", None) is not None:
        fix_pos = torch.tensor(receptacle.pos, dtype=torch.float32, device=env.device)
        return fix_pos.unsqueeze(0).repeat(env.num_envs, 1)

    raise KeyError(
        f"Cannot resolve receptacle position for {receptacle_name!r}; "
        "not in scene assets and no fixture position found."
    )


def _receptacle_xy_half_threshold(env: ManagerBasedEnv, receptacle_name: str) -> float:
    """Half of max(x, y) extent for a receptacle object or fixture."""
    size_x = size_y = None
    receptacle = _resolve_receptacle_fixture(env, receptacle_name)

    if receptacle is not None:
        if receptacle_name in env.cfg.isaaclab_arena_env.task.objects:
            size_x, size_y = receptacle.size[0], receptacle.size[1]
        else:
            fxtr_size = getattr(receptacle, "size", None)
            if fxtr_size is not None and len(fxtr_size) >= 2:
                size_x, size_y = float(fxtr_size[0]), float(fxtr_size[1])
            elif getattr(receptacle, "width", None) is not None and getattr(receptacle, "depth", None) is not None:
                size_x, size_y = float(receptacle.width), float(receptacle.depth)

    if size_x is None or size_y is None:
        raise KeyError(
            f"Cannot resolve receptacle size for {receptacle_name!r}; "
            "not in task.objects and no matching fixture found."
        )
    return float(max(size_x, size_y)) / 2.0


def check_obj_in_receptacle_no_contact(env: ManagerBasedEnv, obj_name: str, receptacle_name: str, th: float = None) -> torch.Tensor:
    """
    check if object is in receptacle object based on threshold
    """
    obj_pos = _scene_asset_pos_w(env, obj_name)
    if obj_pos is None:
        raise KeyError(f"Cannot resolve object position for {obj_name!r}")
    recep_pos = _receptacle_pos_w(env, receptacle_name)
    if th is None:
        th = _receptacle_xy_half_threshold(env, receptacle_name)
    xy_delta = obj_pos[:, :2] - recep_pos[:, :2]
    xy_distance = torch.linalg.vector_norm(xy_delta, dim=-1)
    is_closed = xy_distance < th  # (env_num, )
    return is_closed


def check_obj_fixture_contact(env: ManagerBasedEnv, obj_name: str, fixture_name: str) -> torch.Tensor:
    """
    check if object is in contact with fixture
    """
    obj = env.cfg.isaaclab_arena_env.task.objects[obj_name]
    fixture = env.cfg.isaaclab_arena_env.task.get_fixture(fixture_name)
    return check_contact(env, obj, fixture)


def check_obj_any_counter_contact(env: ManagerBasedEnv, kit_env, obj_name: str) -> torch.Tensor:
    """
    check if the object is in contact with any counter fixture in the environment.
    """
    from robofinals.core.models.fixtures import Counter
    for fixture in kit_env.fixtures.values():
        if isinstance(fixture, Counter):
            if check_obj_fixture_contact(env, obj_name, fixture).any():
                return torch.tensor([True], device=env.device, dtype=torch.bool).repeat(env.num_envs)
    return torch.tensor([False], device=env.device, dtype=torch.bool).repeat(env.num_envs)


def check_fixture_in_receptacle(env: ManagerBasedEnv, fixture_name: str, fixture_object: str, receptacle_name: str, th: float = None) -> torch.Tensor:
    """
    check if fixture is in receptacle object based on threshold
    """
    is_contact = check_obj_fixture_contact(env, receptacle_name, fixture_name)

    recep = env.cfg.isaaclab_arena_env.task.objects[receptacle_name]

    fix_pos = torch.tensor(env.scene.articulations[fixture_object].data.root_link_pos_w[..., 0:2])
    recep_pos = body_com_pos_w_mean_over_bodies(env.scene.rigid_objects[receptacle_name].data)  # (env_num, 3)
    if th is None:
        th = recep.horizontal_radius * 0.7
    is_closed = torch.norm(fix_pos[:, :2] - recep_pos[:, :2], dim=-1) < th  # (env_num, )
    return is_contact & is_closed


def check_obj_grasped(env, obj_name, threshold=0.035):
    """
    Check if the gripper has grasped the object by analyzing contact and proximity
    """
    robot_articulation = env.scene.articulations["robot"].data.joint_names
    if "right_hand_midddle_0_joint" in robot_articulation:
        gripper_joints = ["right_hand_middle_0_joint", "right_hand_middle_1_joint"]
    elif "right_gripper1" in robot_articulation:
        gripper_joints = ["right_gripper1", "right_gripper2"]
    elif any("finger_joint" in name for name in robot_articulation):
        gripper_joints = [name for name in robot_articulation if "finger_joint" in name]

    joint_index = [robot_articulation.index(joint) for joint in gripper_joints]
    gripper_joint_positions = _asset_data_to_torch(env.scene.articulations["robot"].data.joint_pos)[:, joint_index]
    gripper_closed = (gripper_joint_positions < threshold).all()
    is_contact = torch.tensor([False], dtype=torch.bool, device=env.device).repeat(env.num_envs)
    for gripper_name in [name for name in list(env.scene.sensors.keys()) if "gripper" in name and "contact" in name]:
        is_contact |= check_contact(env, gripper_name.replace("_contact", ""), env.cfg.isaaclab_arena_env.task.objects[obj_name])
    return gripper_closed & is_contact


def gripper_obj_far(env, obj_name="obj", th=0.25, eef_name=None, force_th=0.1) -> torch.Tensor:
    """
    check if gripper is far from object based on distance defined by threshold
    """

    if not isinstance(obj_name, str):
        obj_name = obj_name.name

    if obj_name in env.scene.rigid_objects:
        obj_pos = body_com_pos_w_for_gripper_distance(env.scene.rigid_objects[obj_name].data)
    elif obj_name in env.scene.articulations:
        articulation_obj = env.scene.articulations[obj_name]
        obj_pos = body_com_pos_w_for_gripper_distance(articulation_obj.data)
    else:
        available_rigid = tuple(env.scene.rigid_objects.keys())
        available_articulations = tuple(env.scene.articulations.keys())
        raise KeyError(
            f"Object {obj_name!r} was not found in scene rigid_objects or articulations. "
            f"Available rigid_objects={available_rigid}, articulations={available_articulations}."
        )

    target_pos_w = _asset_data_to_torch(env.scene["ee_frame"].data.target_pos_w)
    if target_pos_w.ndim == 2:
        target_pos_w = target_pos_w.unsqueeze(1)
    elif target_pos_w.ndim != 3:
        raise RuntimeError(
            f"ee_frame.target_pos_w: unexpected ndim={target_pos_w.ndim} "
            f"shape={tuple(target_pos_w.shape)}"
        )
    if eef_name is None:
        gripper_site_pos = target_pos_w  # (num_envs, num_eefs, 3)
    else:
        eef_frame_data = env.scene["ee_frame"].data
        eef_index = eef_frame_data.target_frame_names.index(eef_name)
        gripper_site_pos = target_pos_w[:, eef_index: eef_index + 1, :]  # (num_envs, 1, 3)

    # gripper_site_pos: (num_envs, num_eefs, 3)
    # obj_pos: (num_envs, num_bodies, 3)

    gripper_expanded = gripper_site_pos.unsqueeze(2)  # (num_envs, num_eefs, 1, 3)
    obj_expanded = obj_pos.unsqueeze(1)  # (num_envs, 1, num_bodies, 3)

    # Compute distance to all bodies: (num_envs, num_eefs, num_bodies)
    distances = torch.linalg.vector_norm(gripper_expanded - obj_expanded, dim=-1)

    # Distance check: gripper must be far from all body parts
    distance_check = torch.all(distances > th, dim=-1)  # (num_envs, num_eefs)
    distance_far = torch.all(distance_check, dim=-1)  # (num_envs,)

    # Contact force check: no significant contact force
    force_check = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    # Check left gripper contact force
    if "left_gripper_contact" in env.scene.sensors:
        force_check &= env.backend_runtime.contacts.force_magnitude("left_gripper_contact") < force_th

    # Check right gripper contact force
    if "right_gripper_contact" in env.scene.sensors:
        force_check &= env.backend_runtime.contacts.force_magnitude("right_gripper_contact") < force_th

    # Combined check: both distance and force conditions must be satisfied
    return distance_far & force_check


def gripper_fixture_far(env: ManagerBasedEnv, fixture_object: str, th: float = 0.2) -> torch.Tensor:
    """
    check if gripper is far from fixture based on distance defined by threshold

    Args:
        env (Env): environment
        fixture_object: fixture object
        th (float): threshold for distance

    Returns:
        torch.Tensor: (num_envs,) - True if gripper is far from fixture for each environment
    """
    # Get fixture position for all environments
    fix_pos = _asset_data_to_torch(env.scene.articulations[fixture_object].data.root_link_pos_w)[..., 0:2]

    # Get gripper position for all environments
    gripper_site_pos = _asset_data_to_torch(env.scene["ee_frame"].data.target_pos_w)

    # Calculate distance between gripper and fixture for all environments
    gripper_fixture_dist = torch.norm(gripper_site_pos[:, :, :2] - fix_pos.unsqueeze(1), dim=-1)  # (num_envs, num_eefs)

    # Check if gripper is far from fixture for all end effectors
    gripper_fixture_far = gripper_fixture_dist > th  # (num_envs, num_eefs)

    # Return True only if ALL end effectors are far from fixture
    return torch.all(gripper_fixture_far, dim=-1)  # (num_envs,)


def obj_cos(env: ManagerBasedEnv, obj_name: str = "obj", ref: Tuple[float, float, float] = (0, 0, 1)) -> float:
    def cos(u, v):
        return np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-10)

    quat_t = body_com_quat_w_mean_over_bodies(env.scene.rigid_objects[obj_name].data)
    obj_quat = quat_t[0]
    obj_quat_np = obj_quat.detach().cpu().numpy()
    obj_mat = T.quat2mat(obj_quat_np)
    return cos(obj_mat[:, 2], np.array(ref))


def get_obj_lang(env: ManagerBasedEnv, obj_name: str = "obj", get_preposition: bool = False) -> Tuple[str, str]:
    """
    gets a formatted language string for the object (replaces underscores with spaces)

    Args:
        obj_name (str): name of object
        get_preposition (bool): if True, also returns preposition for object

    Returns:
        str: language string for object
    """
    obj_cfg = None
    for cfg in env.object_cfgs:
        if cfg["name"] == obj_name:
            obj_cfg = cfg
            break
    lang = obj_cfg["info"]["category"]

    # replace some phrases
    if lang == "kettle electric":
        lang = "electric kettle"
    elif lang == "kettle non electric":
        lang = "kettle"
    elif lang == "bread_flat":
        lang = "bread"

    if not get_preposition:
        return lang

    if lang in ["bowl", "pot", "pan"]:
        preposition = "in"
    elif lang in ["plate"]:
        preposition = "on"
    else:
        raise ValueError("obj category not in bowl / pot / pan / plate")

    return lang, preposition


def quat2mat(quaternion):
    """
    Converts given quaternion to matrix.

    Args:
        quaternion (torch.Tensor): (N,4) tensor of quaternions in (w,x,y,z) format

    Returns:
        torch.Tensor: (N,3,3) tensor of rotation matrices
    """
    q = quaternion.to(dtype=torch.float32)

    n = torch.sum(q * q, dim=-1, keepdim=True)
    mask = (n < 1e-7).squeeze(-1)

    q = torch.where(mask.unsqueeze(-1), torch.tensor([1., 0., 0., 0.], device=q.device), q)
    n = torch.where(mask.unsqueeze(-1), torch.ones_like(n), n)

    q *= torch.sqrt(2.0 / n)
    q2 = q.unsqueeze(-1) @ q.unsqueeze(-2)

    mat = torch.stack([
        torch.stack([1.0 - q2[..., 2, 2] - q2[..., 3, 3], q2[..., 1, 2] - q2[..., 3, 0], q2[..., 1, 3] + q2[..., 2, 0]], dim=-1),
        torch.stack([q2[..., 1, 2] + q2[..., 3, 0], 1.0 - q2[..., 1, 1] - q2[..., 3, 3], q2[..., 2, 3] - q2[..., 1, 0]], dim=-1),
        torch.stack([q2[..., 1, 3] - q2[..., 2, 0], q2[..., 2, 3] + q2[..., 1, 0], 1.0 - q2[..., 1, 1] - q2[..., 2, 2]], dim=-1)
    ], dim=-2)

    return torch.where(mask.unsqueeze(-1).unsqueeze(-1),
                       torch.eye(3, dtype=torch.float32, device=q.device).expand(q.shape[0], -1, -1),
                       mat)


def construct_full_env_path(env_regex_ns_template: str, env_index: int, fixture_name: str) -> str:

    return os.path.join(f"{env_regex_ns_template.rstrip('.*')}{env_index}", "Scene", fixture_name)


def check_object_stable(env: ManagerBasedEnv, obj: str, threshold: float = 0.5) -> torch.Tensor:
    '''
    check if the object is stable
    '''
    obj_vel = body_com_vel_w_per_env_first_link(env.scene.rigid_objects[obj].data)
    obj_vel_norm = torch.norm(obj_vel, dim=-1)
    return obj_vel_norm < threshold


def project_point_to_segment(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Projects a point onto a line segment, and clamps it to the segment bounds.

    Args:
        point (np.ndarray): The point to project, shape (2,) or (3,)
        seg_start (np.ndarray): Start point of the segment, same shape as point
        seg_end (np.ndarray): End point of the segment, same shape as point

    Returns:
        np.ndarray: The closest point on the segment to the input point
        float: The distance to the segment

    From ChatGPT
    """
    # Convert to numpy arrays in case they aren't already
    point = np.array(point)
    seg_start = np.array(seg_start)
    seg_end = np.array(seg_end)

    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)

    if seg_len_sq == 0.0:
        # Segment is a point
        return seg_start, 0.0

    # Compute t: projection factor along the segment
    t = np.dot(point - seg_start, seg_vec) / seg_len_sq
    t_clamped = np.clip(t, 0.0, 1.0)

    # Compute the closest point
    closest_point = seg_start + t_clamped * seg_vec
    dist = np.linalg.norm(point - closest_point)

    return closest_point, dist


def check_place_obj1_on_obj2(env: ManagerBasedEnv, obj1: str, obj2: str, th_z_axis_cos: float = 0.8, th_xy_dist: float = 0.25, th_xyz_vel: float = 0.5, gipper_th: float = 0.25) -> dict:
    """
    check if obj1 is placed on obj2
    obj1 and obj2 must be a fixture moveable or a object

    Args:
        env (Env): environment
        obj1 : name of object 1 or a moveable fixture
        obj2 : name of object 2 or a moveable fixture
        th_z_axis_cos (float): threshold for z-axis cosine similarity
        th_xy_dist (float): threshold for xy distance
        th_xyz_vel (float): threshold for xyz velocity
        gipper_th (float): threshold for gripper distance

    Returns:
        dict: success state of the task with tensor values for multi-env support
        - gripper_far: check if gripper is far from obj1 and obj2 (tensor)
        - obj1_is_standing: check if obj1 is standing (tensor)
        - obj1_in_obj2: check if obj1 is in obj2 (tensor)
        - obj1_stable: check if obj1 is stable (tensor)
        - obj1_contact_with_obj2: check if obj1 is in contact with obj2 (tensor)

    """
    import torch

    # Get object positions - use torch.mean for multi-body objects like check_obj_in_receptacle
    if obj1 in env.cfg.isaaclab_arena_env.task.objects:
        if obj1 in env.scene.rigid_objects:
            obj1_obj = env.cfg.isaaclab_arena_env.task.objects[obj1]
            obj1_pos = body_com_pos_w_mean_over_bodies(env.scene.rigid_objects[obj1].data)  # (num_envs, 3)
            obj1_vel = body_com_vel_w_mean_over_bodies(env.scene.rigid_objects[obj1].data)  # (num_envs, 3)
            obj1_quat = body_com_quat_w_mean_over_bodies(env.scene.rigid_objects[obj1].data)  # (num_envs, 4)
            obj1_rot_mat = matrix_from_quat(obj1_quat)  # (num_envs, 3, 3)
        elif obj1 in env.scene.articulations:
            obj1_obj = env.cfg.isaaclab_arena_env.task.objects[obj1]
            obj1_pos = body_com_pos_w_mean_over_bodies(env.scene.articulations[obj1].data)  # (num_envs, 3)
            obj1_vel = body_com_vel_w_mean_over_bodies(env.scene.articulations[obj1].data)  # (num_envs, 3)
            obj1_quat = body_com_quat_w_mean_over_bodies(env.scene.articulations[obj1].data)  # (num_envs, 4)
            obj1_rot_mat = matrix_from_quat(obj1_quat)  # (num_envs, 3, 3)
    else:
        obj1_name = obj1 if isinstance(obj1, str) else obj1.name
        obj1_pos = env.scene.state['articulation'][obj1_name]['root_pose'][:, :3]  # (num_envs, 3)
        obj1_quat = env.scene.state['articulation'][obj1_name]['root_pose'][:, 3:]  # (num_envs, 4)
        obj1_rot_mat = matrix_from_quat(obj1_quat)  # (num_envs, 3, 3)
        obj1_vel = env.scene.state['articulation'][obj1_name]['root_velocity'][:, :3]  # (num_envs, 3)

    if obj2 in env.cfg.isaaclab_arena_env.task.objects:
        if obj2 in env.scene.rigid_objects:
            obj2_obj = env.cfg.isaaclab_arena_env.task.objects[obj2]
            obj2_pos = body_com_pos_w_mean_over_bodies(env.scene.rigid_objects[obj2].data)  # (num_envs, 3)
        elif obj2 in env.scene.articulations:
            obj2_obj = env.cfg.isaaclab_arena_env.task.objects[obj2]
            obj2_pos = body_com_pos_w_mean_over_bodies(env.scene.articulations[obj2].data)  # (num_envs, 3)
    else:
        obj2_obj = obj2
        obj2_name = obj2 if isinstance(obj2, str) else obj2.name
        obj2_pos = env.scene.state['articulation'][obj2_name]['root_pose'][:, :3]  # (num_envs, 3)

    # Calculate z-axis cosine similarity for all environments
    obj1_z_axis = obj1_rot_mat[:, :, 2]  # (num_envs, 3)
    world_z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=obj1_z_axis.dtype)
    z_axis_cos = torch.abs(torch.sum(obj1_z_axis * world_z_axis, dim=-1))  # (num_envs,)

    # Calculate xy distance for all environments - similar to check_obj_in_receptacle
    xy_dist = torch.norm(obj1_pos[:, :2] - obj2_pos[:, :2], dim=-1)  # (num_envs,)
    obj1_obj2_size_xy_min = min(obj2_obj.size[:2])

    # Calculate velocity magnitude for all environments
    xyz_vel = torch.norm(obj1_vel, dim=-1)  # (num_envs,)

    # Check gripper distance for all environments
    gripper_far_obj1 = gripper_obj_far(env, obj1, th=gipper_th)  # (num_envs,)
    gripper_far_obj2 = gripper_obj_far(env, obj2, th=gipper_th)  # (num_envs,)
    gripper_far = gripper_far_obj1 & gripper_far_obj2  # (num_envs,)

    # Calculate success conditions for all environments
    obj1_is_standing = z_axis_cos > th_z_axis_cos  # (num_envs,)
    obj1_in_obj2 = xy_dist < obj1_obj2_size_xy_min * th_xy_dist  # (num_envs,)
    obj1_stable = xyz_vel < th_xyz_vel  # (num_envs,)
    return gripper_far & obj1_is_standing & obj1_in_obj2 & obj1_stable


def get_object_pos(env, obj_name):
    if obj_name in env.scene.articulations:
        return body_com_pos_w_per_env_first_link(env.scene.articulations[obj_name].data)
    elif obj_name in env.scene.rigid_objects:
        return body_com_pos_w_per_env_first_link(env.scene.rigid_objects[obj_name].data)


def check_near(obj1_pos, obj2_pos, th=0.2):
    dis = torch.norm(obj1_pos - obj2_pos, dim=-1)
    return dis < th


def check_place_obj1_side_by_obj2(env: ManagerBasedEnv, obj1: str, obj2: str, check_states: dict, gipper_th: float = 0.25) -> torch.Tensor:
    """
    check if obj1 is placed side by side with obj2

    Args:
        env (Env): environment
        obj1 (str): name of object 1
        obj2 (str): name of object 2
        check_states (dict): check states
        check_states = {
            "gripper_far": True,   # obj1 and obj2 should be far from the gripper
            "contact": False,   # obj1 should not be in contact with obj2
            "side": 'right',    # relative position of obj1 to obj2, left, right, front, back
            "side_threshold": 0.25,    # threshold for distance between obj1 and obj2 in other directions, 0.25*(min(obj2_obj.size[:2]) + min(obj1_obj.size[:2]))/ 2
            "margin_threshold": [0.001, 0.1],    # threshold for distance between obj1 and obj2, 0.001 for min distance, 0.1 for max distance
            "parallel": [0, 1, 1],    # whether obj1 is parallel to the world axis, can be multiple, here means parallel to y and z axis
            "parallel_threshold": 0.95,    # threshold for parallel to world axis
            "stable_threshold": 0.5,    # threshold for stable, velocity vector length less than 0.5
        }

    Returns:
        torch.Tensor: success state of the task for all environments (num_envs,)

    """
    import torch

    # Get object positions - use torch.mean for multi-body objects like check_place_obj1_on_obj2
    if obj1 in env.cfg.isaaclab_arena_env.task.objects:
        obj1_obj = env.cfg.isaaclab_arena_env.task.objects[obj1]
        obj1_pos = body_com_pos_w_mean_over_bodies(env.scene.rigid_objects[obj1].data)  # (num_envs, 3)
        obj1_vel = body_com_vel_w_mean_over_bodies(env.scene.rigid_objects[obj1].data)  # (num_envs, 3)
        obj1_quat = body_com_quat_w_mean_over_bodies(env.scene.rigid_objects[obj1].data)  # (num_envs, 4)
        obj1_rot_mat = matrix_from_quat(obj1_quat)  # (num_envs, 3, 3)
    else:
        obj1_obj = obj1
        obj1_name = obj1 if isinstance(obj1, str) else obj1.name
        obj1_pos = env.scene.state['articulation'][obj1_name]['root_pose'][:, :3]  # (num_envs, 3)
        obj1_quat = env.scene.state['articulation'][obj1_name]['root_pose'][:, 3:]  # (num_envs, 4)
        obj1_rot_mat = matrix_from_quat(obj1_quat)  # (num_envs, 3, 3)
        obj1_vel = env.scene.state['articulation'][obj1_name]['root_velocity'][:, :3]  # (num_envs, 3)

    if obj2 in env.cfg.isaaclab_arena_env.task.objects:
        obj2_obj = env.cfg.isaaclab_arena_env.task.objects[obj2]
        obj2_pos = body_com_pos_w_mean_over_bodies(env.scene.rigid_objects[obj2].data)  # (num_envs, 3)
    else:
        obj2_obj = obj2
        obj2_name = obj2 if isinstance(obj2, str) else obj2.name
        obj2_pos = env.scene.state['articulation'][obj2_name]['root_pose'][:, :3]  # (num_envs, 3)

    # Initialize all conditions as True for environments
    all_conditions = []

    # Check gripper distance for all environments
    if 'gripper_far' in check_states:
        gripper_far_obj1 = gripper_obj_far(env, obj1, th=gipper_th)  # (num_envs,)
        gripper_far_obj2 = gripper_obj_far(env, obj2, th=gipper_th)  # (num_envs,)
        gripper_far = gripper_far_obj1 & gripper_far_obj2  # (num_envs,)
        all_conditions.append(gripper_far)

    # Check contact for all environments
    if 'contact' in check_states:
        need_contact = check_states['contact']
        contact_result = check_contact(env, obj1_obj, obj2_obj)  # (num_envs,)
        if need_contact:
            all_conditions.append(contact_result)
        else:
            all_conditions.append(~contact_result)

    # Check stability for all environments
    if 'stable_threshold' in check_states:
        xyz_vel = torch.norm(obj1_vel, dim=-1)  # (num_envs,)
        stable = xyz_vel < check_states['stable_threshold']  # (num_envs,)
        all_conditions.append(stable)

    # Check parallel alignment for all environments
    if 'parallel' in check_states:
        parallel_conditions = []
        for index, axis in enumerate(check_states['parallel']):
            if axis:
                # Calculate parallel alignment for all environments
                obj_axis = obj1_rot_mat[:, :, index]  # (num_envs, 3)
                world_axis = [0, 0, 0]
                world_axis[index] = 1
                world_axis_tensor = torch.tensor(world_axis, device=env.device, dtype=obj_axis.dtype)
                axis_cos = torch.abs(torch.sum(obj_axis * world_axis_tensor, dim=-1))  # (num_envs,)
                parallel = axis_cos > check_states.get('parallel_threshold', 0.9)  # (num_envs,)
                parallel_conditions.append(parallel)
        if parallel_conditions:
            # All parallel conditions must be satisfied
            all_parallel = torch.stack(parallel_conditions, dim=0)
            all_conditions.append(torch.all(all_parallel, dim=0))

    # Check side positioning for all environments
    if 'side' in check_states:
        side = check_states['side']
        obj1_obj2_size_xy_min = (min(obj2_obj.size[:2]) + min(obj1_obj.size[:2])) / 2
        obj1_obj2_center_xydist = torch.norm(obj1_pos[:, :2] - obj2_pos[:, :2], dim=-1)  # (num_envs,)
        obj1_obj2_edge_xydist = obj1_obj2_center_xydist - obj1_obj2_size_xy_min  # (num_envs,)

        margin_threshold = check_states.get('margin_threshold', [0, 0])
        margin_threshold_min = margin_threshold[0]
        margin_threshold_max = margin_threshold[1]
        space_success = (obj1_obj2_edge_xydist > margin_threshold_min) & (obj1_obj2_edge_xydist < margin_threshold_max)  # (num_envs,)

        if side == "left":
            y_dist = torch.abs(obj1_pos[:, 1] - obj2_pos[:, 1])  # (num_envs,)
            side_condition = (obj1_pos[:, 0] > obj2_pos[:, 0]) & (y_dist < obj1_obj2_size_xy_min * check_states.get('side_threshold', 0.25))  # (num_envs,)
            all_conditions.append(space_success & side_condition)
        elif side == "right":
            y_dist = torch.abs(obj1_pos[:, 1] - obj2_pos[:, 1])  # (num_envs,)
            side_condition = (obj1_pos[:, 0] < obj2_pos[:, 0]) & (y_dist < obj1_obj2_size_xy_min * check_states.get('side_threshold', 0.25))  # (num_envs,)
            all_conditions.append(space_success & side_condition)
        elif side == "back":
            x_dist = torch.abs(obj1_pos[:, 0] - obj2_pos[:, 0])  # (num_envs,)
            side_condition = (obj1_pos[:, 1] > obj2_pos[:, 1]) & (x_dist < obj1_obj2_size_xy_min * check_states.get('side_threshold', 0.25))  # (num_envs,)
            all_conditions.append(space_success & side_condition)
        elif side == "front":
            x_dist = torch.abs(obj1_pos[:, 0] - obj2_pos[:, 0])  # (num_envs,)
            side_condition = (obj1_pos[:, 1] < obj2_pos[:, 1]) & (x_dist < obj1_obj2_size_xy_min * check_states.get('side_threshold', 0.25))  # (num_envs,)
            all_conditions.append(space_success & side_condition)

    # Combine all conditions with AND operation
    if all_conditions:
        return torch.all(torch.stack(all_conditions, dim=0), dim=0)  # (num_envs,)
    else:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)


def check_obj_location_on_stove(env: ManagerBasedEnv, stove: Fixture, obj_name: str, threshold: float = 0.08, need_knob_on: bool = True) -> List[Tuple[str, bool]]:
    """
    Check if the object is on the stove and close to a burner and the knob is on (optional).
    Returns the location of the burner if the object is on the stove, close to a burner, and the burner is on (optional).
    None otherwise.
    """

    knobs_state = stove.get_knobs_state(env=env)
    obj_pos = body_com_pos_w_per_env_first_link(env.scene.rigid_objects[obj_name].data)
    obj_on_stove = check_obj_fixture_contact(env, obj_name, stove)
    stove_pos = torch.tensor(stove.pos, device=env.device)
    stove_rot = T.euler2mat(torch.tensor([0.0, 0.0, stove.rot], device=env.device)).to(dtype=torch.float32)
    locations = []
    for env_id in range(len(obj_on_stove)):
        found_location = False
        if obj_on_stove[env_id]:
            for location, site in stove.burner_sites.items():
                if site[env_id] is not None:
                    burner_pos = stove_rot @ torch.tensor(site[env_id].GetAttribute("xformOp:translate").Get(), device=env.device) + stove_pos + env.scene.env_origins[env_id]
                    dist = torch.norm(burner_pos[:2] - obj_pos[env_id][:2])
                    obj_on_site = dist < threshold
                    knob_on = (
                        (0.35 <= torch.abs(knobs_state[location]) <= 2 * torch.pi - 0.35)
                        if location in knobs_state
                        else False
                    )
                    check_result = obj_on_site if not need_knob_on else obj_on_site and knob_on
                    if check_result:
                        found_location = True
                        locations.append(location)
                        break
        if not found_location:
            locations.append(None)
    return locations


def is_obj_z_up(env: ManagerBasedEnv, obj_name: str = "obj", th: float = 5.0) -> torch.Tensor:
    obj_quat = sim_quat_raw_to_xyzw_torch(_asset_data_to_torch(env.scene[obj_name].data.root_quat_w))
    r, p, y = euler_xyz_from_quat(obj_quat)
    th = th * torch.pi / 180.0
    return torch.logical_and(torch.abs(r) < th, torch.abs(p) < th)


def grasp_obj(env, obj_name='obj', threshold=0.15):
    close_to_obj = torch.logical_not(gripper_obj_far(env, obj_name, th=threshold, eef_name="tool_left_arm"))
    left_hand_action_closed = env.action_manager.action[:, -2] > 0.5
    return torch.logical_and(close_to_obj, left_hand_action_closed)


def put_obj_to_coffee_machine(env: ManagerBasedEnv, obj_name: str = "obj", judge_obj_in_coffee_machine: callable = None) -> torch.Tensor:
    obj_height = body_com_pos_w_per_env_first_link(env.scene.rigid_objects[obj_name].data)[:, 2]
    default_obj_height = _asset_data_to_torch(env.scene.rigid_objects[obj_name].data.default_root_state)[:, 2]
    higher_than_default = obj_height > default_obj_height

    close_to_obj = torch.logical_not(gripper_obj_far(env, obj_name, th=0.15, eef_name="tool_left_arm"))
    left_hand_action_open = env.action_manager.action[:, -2] < 0.5
    try:
        obj_in_coffee_machine = judge_obj_in_coffee_machine(env, obj_name, xy_thresh=0.09)
    except Exception:
        obj_in_coffee_machine = torch.tensor([False] * env.num_envs, dtype=torch.bool, device=env.device)

    return torch.logical_and(close_to_obj, left_hand_action_open) & obj_in_coffee_machine & higher_than_default


def obj_fixture_bbox_min_dist(env: ManagerBasedEnv, obj_name: str, fixture: Fixture) -> torch.Tensor:
    """
    Gets the minimum distance between a fixture and an object by computing the minimal axis-aligned bounding separation.
    """
    fix_pts = fixture.get_ext_sites(all_points=True, relative=False)
    fix_coords = np.array(fix_pts)
    fix_min = fix_coords.min(axis=0)
    fix_max = fix_coords.max(axis=0)

    all_sep_distances = []

    for i in range(env.cfg.scene.num_envs):
        fix_min_i = fix_min + env.scene.env_origins[i].cpu().numpy()
        fix_max_i = fix_max + env.scene.env_origins[i].cpu().numpy()
        if obj_name in env.scene.rigid_objects:
            pos_t = _asset_data_to_torch(env.scene.rigid_objects[obj_name].data.body_com_pos_w)
            quat_t = _asset_data_to_torch(env.scene.rigid_objects[obj_name].data.body_com_quat_w)
            obj_pos = (pos_t[i, 0, :] if pos_t.ndim == 3 else pos_t[i, :]).cpu().numpy()
            obj_quat = sim_quat_raw_to_xyzw_np(
                (quat_t[i, 0, :] if quat_t.ndim == 3 else quat_t[i, :]).cpu().numpy()
            )
        elif obj_name in env.scene.articulations:
            pos_t = _asset_data_to_torch(env.scene.articulations[obj_name].data.body_com_pos_w)
            quat_t = _asset_data_to_torch(env.scene.articulations[obj_name].data.body_com_quat_w)
            obj_pos = (pos_t[i, 0, :] if pos_t.ndim == 3 else pos_t[i, :]).cpu().numpy()
            obj_quat = sim_quat_raw_to_xyzw_np(
                (quat_t[i, 0, :] if quat_t.ndim == 3 else quat_t[i, :]).cpu().numpy()
            )
        obj = env.cfg.isaaclab_arena_env.task.objects[obj_name]
        obj_pts = obj.get_bbox_points(trans=obj_pos, rot=obj_quat)
        obj_coords = np.array(obj_pts)
        obj_min = obj_coords.min(axis=0)
        obj_max = obj_coords.max(axis=0)

        sep = np.zeros(3)
        for j, axis in enumerate(["x", "y", "z"]):
            if fix_max_i[j] < obj_min[j]:
                sep[j] = obj_min[j] - fix_max_i[j]
            elif obj_max[j] < fix_min_i[j]:
                sep[j] = fix_min_i[j] - obj_max[j]
            else:
                sep[j] = 0.0

        sep_distance = np.linalg.norm(sep)
        all_sep_distances.append(sep_distance)

    return torch.tensor(all_sep_distances, dtype=torch.float32, device=env.device)


def check_contact(env: ManagerBasedEnv, geoms_1: str | USDObject | Fixture, geoms_2: str | USDObject | Fixture) -> torch.Tensor:
    """Backend-neutral pair contact query."""
    return env.backend_runtime.contacts.check_contact(geoms_1, geoms_2)


def calculate_contact_force(env: ManagerBasedEnv, geom: str | USDObject | Fixture) -> torch.Tensor:
    """
    calculate the contact force on the geom
    """
    if f"{geom}_contact" not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    return env.backend_runtime.contacts.force_magnitude(f"{geom}_contact")


def set_obj_rgb(env, obj_name, rgb_value):
    """
    Set the RGB color of an object in the USD stage.
    """
    obj = env.cfg.isaaclab_arena_env.task.objects[obj_name]
    obj_usd = usd(obj.obj_path)
    obj_usd.set_rgb(obj_usd.stage.GetPseudoRoot(), rgb=rgb_value)


def add_obj_liquid_site(env, obj_name, liquid_rgba):
    """
    Set the liquid site rgb for an object in the USD stage.
    """
    obj = env.cfg.isaaclab_arena_env.task.objects[obj_name]
    obj_usd = usd(obj.obj_path)
    site_prims = obj_usd.get_all_prims(obj_usd.stage)
    liquid_sites = [site for site in site_prims if site is not None and site.IsValid() and "liquid" in site.GetName()]
    Site_prim = obj_usd.get_prim_by_name(prim=obj_usd.stage.GetPseudoRoot(), name="Sites", only_xform=False)[0]
    for site in liquid_sites:
        Site_prim.GetAttribute("visibility").Set("inherited")
        Site_prim.GetAttribute("purpose").Set("default")
        site.GetAttribute("visibility").Set("inherited")
        site.GetAttribute("purpose").Set("default")
        site.GetAttribute("primvars:displayColor").Set(liquid_rgba)  # liquid_rgba is a list of 3 floats,such as [(0.0, 0.0, 1.0)],which means blue color
