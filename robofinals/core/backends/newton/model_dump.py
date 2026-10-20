# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""JSON dumps for robofinals Newton models."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def dump_newton_model(
    model,
    *,
    usd_path: Path | str,
    solver_name: str,
    output_dir: str | Path = "model_dumps",
    state=None,
    solver=None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Dump a finalized Newton model to a timestamped JSON file.

    Args:
        model: Finalized Newton model.
        usd_path: Source USD path used to build the model.
        solver_name: Solver selected for this run.
        output_dir: Directory where the JSON file is written.
        state: Optional Newton state to include in summaries.
        solver: Optional solver to summarize.
        extra: Additional JSON-serializable metadata.

    Returns:
        Path to the written JSON file.
    """
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{_safe_name(Path(usd_path).stem)}_{timestamp}.json"
    output_path = output_dir / filename

    body_labels = _string_list(getattr(model, "body_label", None))
    shape_labels = _string_list(getattr(model, "shape_label", None))
    joint_labels = _string_list(getattr(model, "joint_label", None))
    shapes = _shape_summary(model, body_labels, shape_labels)
    bodies = _body_summary(model, state, body_labels, shapes)
    joints = _joint_summary(model, body_labels, joint_labels)

    data = {
        "source": {
            "usd_path": str(Path(usd_path)),
            "solver": solver_name,
            "dumped_at": datetime.now().isoformat(timespec="seconds"),
        },
        "counts": {
            "body_count": int(getattr(model, "body_count", len(body_labels))),
            "shape_count": int(getattr(model, "shape_count", len(shape_labels))),
            "joint_count": int(getattr(model, "joint_count", len(joint_labels))),
            "particle_count": int(getattr(model, "particle_count", 0)),
            "articulation_count": int(getattr(model, "articulation_count", 0)),
        },
        "summary": {
            "bodies": bodies,
            "shapes": shapes,
            "joints": joints,
        },
        "model": _dump_public_attributes(model),
        "state": _dump_state(state),
        "solver": _dump_solver(solver),
        "extra": extra or {},
    }
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _dump_public_attributes(obj) -> dict[str, Any]:
    data = {}
    for name in sorted(dir(obj)):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception as exc:
            data[name] = {"__error__": str(exc)}
            continue
        if callable(value):
            continue
        data[name] = _json_value(value)
    return data


def _dump_state(state) -> dict[str, Any]:
    if state is None:
        return {}
    return _dump_public_attributes(state)


def _dump_solver(solver) -> dict[str, Any]:
    if solver is None:
        return {}
    data = {"type": type(solver).__name__}
    if hasattr(solver, "get_max_contact_count"):
        try:
            data["max_contacts"] = int(solver.get_max_contact_count())
        except Exception as exc:
            data["max_contacts_error"] = str(exc)
    for attr_name in ("mj_model", "mj_data", "mjw_data"):
        if hasattr(solver, attr_name):
            try:
                data[attr_name] = _json_value(getattr(solver, attr_name))
            except Exception as exc:
                data[attr_name] = {"__error__": str(exc)}
    return data


def _shape_summary(model, body_labels: list[str], shape_labels: list[str]) -> list[dict[str, Any]]:
    shape_body = _array_data(getattr(model, "shape_body", None))
    shape_type = _array_data(getattr(model, "shape_type", None))
    shape_flags = _array_data(getattr(model, "shape_flags", None))
    shape_collision_group = _array_data(getattr(model, "shape_collision_group", None))
    shape_transform = _array_data(getattr(model, "shape_transform", None))
    shape_scale = _array_data(getattr(model, "shape_scale", None))
    shape_material_ke = _array_data(getattr(model, "shape_material_ke", None))
    shape_material_kd = _array_data(getattr(model, "shape_material_kd", None))
    shape_material_mu = _array_data(getattr(model, "shape_material_mu", None))
    shape_world = _array_data(getattr(model, "shape_world", None))

    shapes = []
    for index, label in enumerate(shape_labels):
        body_index = int(_safe_index(shape_body, index, -1))
        shapes.append(
            {
                "index": index,
                "label": label,
                "body_index": body_index,
                "body_label": body_labels[body_index] if 0 <= body_index < len(body_labels) else None,
                "type": _safe_index(shape_type, index),
                "flags": _safe_index(shape_flags, index),
                "collision_group": _safe_index(shape_collision_group, index),
                "transform": _safe_index(shape_transform, index),
                "scale": _safe_index(shape_scale, index),
                "material_ke": _safe_index(shape_material_ke, index),
                "material_kd": _safe_index(shape_material_kd, index),
                "material_mu": _safe_index(shape_material_mu, index),
                "world": _safe_index(shape_world, index),
            }
        )
    return shapes


def _body_summary(model, state, body_labels: list[str], shapes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    del model
    body_q = _array_data(getattr(state, "body_q", None)) if state is not None else None
    body_qd = _array_data(getattr(state, "body_qd", None)) if state is not None else None
    bodies = []
    for index, label in enumerate(body_labels):
        bodies.append(
            {
                "index": index,
                "label": label,
                "q": _safe_index(body_q, index),
                "qd": _safe_index(body_qd, index),
                "shape_indices": [
                    shape["index"]
                    for shape in shapes
                    if int(shape["body_index"] if shape["body_index"] is not None else -1) == index
                ],
            }
        )
    return bodies


def _joint_summary(model, body_labels: list[str], joint_labels: list[str]) -> list[dict[str, Any]]:
    joint_type = _array_data(getattr(model, "joint_type", None))
    joint_parent = _array_data(getattr(model, "joint_parent", None))
    joint_child = _array_data(getattr(model, "joint_child", None))
    joint_q_start = _array_data(getattr(model, "joint_q_start", None))
    joint_qd_start = _array_data(getattr(model, "joint_qd_start", None))
    joint_dof_dim = _array_data(getattr(model, "joint_dof_dim", None))
    joint_articulation = _array_data(getattr(model, "joint_articulation", None))

    joints = []
    for index, label in enumerate(joint_labels):
        parent = int(_safe_index(joint_parent, index, -1))
        child = int(_safe_index(joint_child, index, -1))
        joints.append(
            {
                "index": index,
                "label": label,
                "type": _safe_index(joint_type, index),
                "parent": parent,
                "parent_label": body_labels[parent] if 0 <= parent < len(body_labels) else None,
                "child": child,
                "child_label": body_labels[child] if 0 <= child < len(body_labels) else None,
                "q_start": _safe_index(joint_q_start, index),
                "qd_start": _safe_index(joint_qd_start, index),
                "dof_dim": _safe_index(joint_dof_dim, index),
                "articulation": _safe_index(joint_articulation, index),
            }
        )
    return joints


def _array_data(value):
    if value is None:
        return None
    if hasattr(value, "numpy"):
        value = value.numpy()
    return _json_value(np.asarray(value))


def _json_value(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if hasattr(value, "numpy"):
        try:
            return _json_value(value.numpy())
        except Exception:
            pass
    return {"__type__": type(value).__name__, "repr": repr(value)}


def _string_list(values) -> list[str]:
    if values is None:
        return []
    return [str(value) for value in values]


def _safe_index(values, index: int, default=None):
    if values is None:
        return default
    if 0 <= index < len(values):
        return values[index]
    return default


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or "model"
