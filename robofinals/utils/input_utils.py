"""Helpers for reading configuration from env.cfg, embodiment attrs, or environment variables.

Convention: env-var name is derived automatically by upper-casing the cfg_name.
  cfg_name="vive_left_tracker"  →  env-var "VIVE_LEFT_TRACKER"
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _cfg_value(env, name: str, default=None):
    """Read a teleop option from env.cfg or embodiment, falling back to *default*."""
    cfg = getattr(env, "cfg", None)
    if cfg is not None and hasattr(cfg, name):
        return getattr(cfg, name)
    embodiment = getattr(getattr(cfg, "isaaclab_arena_env", None), "embodiment", None)
    if embodiment is not None and hasattr(embodiment, name):
        return getattr(embodiment, name)
    return default


def cfg_or_env_flag(env, cfg_name: str, default: bool = False) -> bool:
    """Read a bool from env.cfg / embodiment, then fall back to the env-var UPPER(cfg_name)."""
    value = _cfg_value(env, cfg_name, None)
    if value is not None:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    env_val = os.environ.get(cfg_name.upper())
    if env_val is not None:
        return env_val.strip().lower() in {"1", "true", "yes", "on"}
    return default


def cfg_or_env_str(env, cfg_name: str, default: str = "") -> str:
    """Read a str from env.cfg / embodiment, then fall back to the env-var UPPER(cfg_name)."""
    value = _cfg_value(env, cfg_name, None)
    if value is not None:
        return str(value)
    return os.environ.get(cfg_name.upper(), default)


def cfg_or_env_float(env, cfg_name: str, default: float = 0.0) -> float:
    """Read a float from env.cfg / embodiment, then fall back to the env-var UPPER(cfg_name)."""
    value = _cfg_value(env, cfg_name, None)
    if value is None:
        value = os.environ.get(cfg_name.upper(), default)
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid float for %s / %s: %r. Falling back to %.4f.",
            cfg_name, cfg_name.upper(), value, default,
        )
        return default
