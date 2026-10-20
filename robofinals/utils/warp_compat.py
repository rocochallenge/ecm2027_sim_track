"""Backend-neutral compatibility helpers for Kit's bundled Warp."""

from enum import IntEnum

def ensure_warp_config_compat() -> None:
    """Backfill Warp APIs missing from Kit's bundled omni.warp.core.

    Newton reads newer Warp logging and deterministic-mode settings at import
    time. Pip ``warp-lang`` provides these, but Isaac Sim can still preload an
    older extension copy at runtime.
    """
    try:
        import warp as wp
    except ImportError:
        return

    if not hasattr(wp, "LOG_DEBUG"):
        wp.LOG_DEBUG = 10
    if not hasattr(wp, "LOG_INFO"):
        wp.LOG_INFO = 20
    if not hasattr(wp, "LOG_WARNING"):
        wp.LOG_WARNING = 30
    if not hasattr(wp, "LOG_ERROR"):
        wp.LOG_ERROR = 40
    if not hasattr(wp.config, "log_level"):
        wp.config.log_level = wp.LOG_INFO

    deterministic_mode = getattr(wp, "DeterministicMode", None)
    if deterministic_mode is None:
        deterministic_mode = getattr(wp.config, "DeterministicMode", None)
    if deterministic_mode is None:
        class DeterministicMode(IntEnum):
            NOT_GUARANTEED = 0
            RUN_TO_RUN = 1
            GPU_TO_GPU = 2

        deterministic_mode = DeterministicMode
    if not hasattr(wp, "DeterministicMode"):
        wp.DeterministicMode = deterministic_mode
    if not hasattr(wp.config, "deterministic"):
        wp.config.deterministic = wp.DeterministicMode.NOT_GUARANTEED
    if not hasattr(wp.config, "deterministic_max_records"):
        wp.config.deterministic_max_records = 0

    if not getattr(wp, "_robofinals_deterministic_options_compat", False):
        original_set_module_options = wp.set_module_options

        def set_module_options_compat(options, *args, **kwargs):
            try:
                return original_set_module_options(options, *args, **kwargs)
            except (KeyError, TypeError, ValueError):
                if not isinstance(options, dict) or "deterministic" not in options:
                    raise
                mode = options.get("deterministic")
                max_records = int(options.get("deterministic_max_records", 0) or 0)
                if mode != wp.DeterministicMode.NOT_GUARANTEED or max_records != 0:
                    raise
                downgraded = {
                    key: value
                    for key, value in options.items()
                    if key not in ("deterministic", "deterministic_max_records")
                }
                if not downgraded:
                    return None
                return original_set_module_options(downgraded, *args, **kwargs)

        wp.set_module_options = set_module_options_compat
        wp._robofinals_deterministic_options_compat = True


