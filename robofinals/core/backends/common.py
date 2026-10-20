"""Backend-independent backend selection primitives."""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum


class BackendKind(StrEnum):
    PHYSX = "physx"
    NEWTON = "newton"

    @classmethod
    def parse(
        cls,
        value: str | "BackendKind" | None,
        *,
        default: "BackendKind" = PHYSX,
    ) -> "BackendKind":
        if value is None or not str(value).strip():
            return default if isinstance(default, cls) else cls(str(default))
        normalized = str(value).strip().lower()
        if normalized == "px":
            normalized = cls.PHYSX.value
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(
                f"Unknown physics backend {value!r}; expected physx or newton"
            ) from exc


def resolve_backend_kind(
    explicit: str | BackendKind | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    default: BackendKind = BackendKind.PHYSX,
) -> BackendKind:
    """Resolve a single backend at process startup: explicit > env > default."""

    env = environ if environ is not None else os.environ
    value = explicit if explicit is not None and str(explicit).strip() else env.get(
        "robofinals_PHYSICS_BACKEND"
    )
    return BackendKind.parse(value, default=default)

__all__ = ["BackendKind", "resolve_backend_kind"]
