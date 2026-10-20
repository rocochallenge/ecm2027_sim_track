"""Resolve assets embedded in recorded episode metadata."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple


RecordedAsset = Tuple[str, str, Dict[str, Any]]


def resolve_recorded_asset(object_cfgs: Dict[str, Any]) -> Optional[RecordedAsset]:
    """Return a local recorded USD asset when its path is available.

    Replay metadata can contain an absolute path from another host. The
    canonical path is preferred, with a conservative ``/root`` to current
    home remap for datasets produced in the standard container image.
    """
    raw_path = object_cfgs.get("asset_name")
    if not raw_path:
        return None

    raw_path = str(raw_path)
    candidates = [Path(raw_path)]
    if raw_path.startswith("/root/"):
        candidates.append(Path.home() / raw_path.removeprefix("/root/"))

    path = None
    for candidate in candidates:
        try:
            if candidate.is_file():
                path = candidate
                break
        except OSError:
            continue
    if path is None:
        return None

    name = path.stem
    obj_res = {
        "source": "local",
        "property": {},
        "metadata": {"local": {"scale": 1.0, "exclude": []}},
        "registryName": object_cfgs.get("obj_groups"),
        "fileVersionId": None,
    }
    return str(path), name, obj_res
