"""Read an evaluator-owned environment configuration without importing Isaac."""
import json
from pathlib import Path
from types import SimpleNamespace


DEFAULTS = {
    "scene_backend": "local", "task_backend": "local", "device": "cpu",
    "robot_scale": 1.0, "first_person_view": False, "disable_fabric": False,
    "num_envs": 1, "usd_simplify": False, "video": False, "for_rl": False,
    "variant": "Visual", "concatenate_terms": False, "distributed": False,
    "physics_backend": "physx", "seed": 42, "sources": [], "object_projects": [],
    "execute_mode": "eval", "replay_cfgs": {"add_camera_to_observation": True},
    "enable_full_local_scene": True, "rl": None,
}
REQUIRED = ("task", "robot", "layout")


def load_eval_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        supplied = json.load(handle)
    if not isinstance(supplied, dict):
        raise ValueError("evaluator config must be a JSON object")
    missing = [key for key in REQUIRED if not isinstance(supplied.get(key), str) or not supplied[key]]
    if missing:
        raise ValueError("missing evaluator configuration: " + ", ".join(missing))
    config = {**DEFAULTS, **supplied}
    if isinstance(config["num_envs"], bool) or not isinstance(config["num_envs"], int) or config["num_envs"] < 1:
        raise ValueError("num_envs must be a positive integer")
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    return SimpleNamespace(**config)


def locked_initializer(factory, locked_cfg, launcher_args):
    """Build an initializer which never passes a policy-supplied config to the factory."""
    from copy import deepcopy

    def initialize(client_cfg=None):
        return factory(deepcopy(locked_cfg), launcher_args)

    return initialize
