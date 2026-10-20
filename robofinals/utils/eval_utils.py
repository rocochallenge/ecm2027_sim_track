from pathlib import Path
import json
from typing import Any
import os


def load_initial_states(path: Path | str | None) -> list[list[dict[str, Any]]]:
    if path is None:
        return []
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))

    data = json.loads(path.read_text())
    worlds = data.get("worlds")
    if not isinstance(worlds, list):
        raise ValueError(f"Expected {path} to contain a top-level 'worlds' list")
    return worlds


def clone_scene_state(value: Any) -> Any:
    import torch
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_scene_state(item) for key, item in value.items()}
    return value


def find_scene_state_entry(scene_state: dict[str, Any], object_name: str) -> dict[str, Any]:
    for asset_type in ("rigid_object", "articulation"):
        asset_states = scene_state.get(asset_type, {})
        if object_name in asset_states:
            return asset_states[object_name]

    available_names = []
    for asset_type in ("rigid_object", "articulation"):
        available_names.extend(scene_state.get(asset_type, {}).keys())
    raise KeyError(
        f"Object '{object_name}' was not found in scene state. "
        f"Available state objects: {sorted(available_names)}"
    )


def build_reset_state_from_eval_initial_states(
    base_scene_state: dict[str, Any],
    world_states: list[list[dict[str, Any]]],
    env_ids: list[int],
):
    import torch

    scene_state = clone_scene_state(base_scene_state)
    if len(world_states) != len(env_ids):
        raise ValueError(f"Expected {len(env_ids)} world states, got {len(world_states)}")

    for env_id, world_state in zip(env_ids, world_states):
        for object_state in world_state:
            object_name = object_state["object_name"]
            pose = object_state["pose"]
            pos, quat_xyzw = pose["pos"], pose["quat"]
            if len(pos) != 3 or len(quat_xyzw) != 4:
                raise ValueError(f"Invalid pose for {object_name}: {pose}")

            state_entry = find_scene_state_entry(scene_state, object_name)
            root_pose = state_entry.get("root_pose")
            if root_pose is None:
                raise KeyError(f"Object '{object_name}' has no root_pose in scene state")

            forced_pose = torch.tensor(pos + quat_xyzw, dtype=root_pose.dtype, device=root_pose.device)
            root_pose[env_id] = forced_pose

            root_velocity = state_entry.get("root_velocity")
            if root_velocity is not None:
                root_velocity[env_id] = 0.0

    return scene_state


def world_states_for_batch(
    initial_states: list[list[dict[str, Any]]], batch_idx: int, num_envs: int, test_num: int
) -> list[list[dict[str, Any]]]:
    start = batch_idx * num_envs
    end = min(start + num_envs, test_num, len(initial_states))
    return initial_states[start:end]


def extract_object_world_states(
    scene_state: dict[str, Any],
    object_names: list[str],
    env_ids: list[int],
    scale: float = 1.0,
) -> list[list[dict[str, Any]]]:
    """Extract per-env object poses from a scene state into put_bottles.json format.

    Each returned world is a list of ``{object_name, pose: {pos, quat}, scale}``
    dicts, where ``pos`` is ``[x, y, z]`` and ``quat`` is ``[qx, qy, qz, qw]``
    (xyzw order, matching ``build_reset_state_from_eval_initial_states``).
    """
    worlds: list[list[dict[str, Any]]] = []
    for env_id in env_ids:
        world: list[dict[str, Any]] = []
        for name in object_names:
            state_entry = find_scene_state_entry(scene_state, name)
            root_pose = state_entry.get("root_pose")
            if root_pose is None:
                raise KeyError(f"Object '{name}' has no root_pose in scene state")
            pose = root_pose[env_id]
            pos = [float(v) for v in pose[:3].tolist()]
            quat = [float(v) for v in pose[3:7].tolist()]
            world.append(
                {
                    "object_name": name,
                    "pose": {"pos": pos, "quat": quat},
                    "scale": [float(scale)],
                }
            )
        worlds.append(world)
    return worlds


def save_world_states(worlds: list[list[dict[str, Any]]], path: Path | str) -> None:
    """Write accumulated per-episode world states to ``{"worlds": [...]}`` JSON."""
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"worlds": worlds}, indent=2, ensure_ascii=False))
