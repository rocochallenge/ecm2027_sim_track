"""Deterministic, inspectable environment compilation pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from robofinals.core.runtime.run_plan import ResolvedRunPlan

from .config_validator import validate_run_plan
from .plugin_registry import PluginRegistry


class CompilePhase(StrEnum):
    RESOLVE = "resolve"
    INSTANTIATE = "instantiate"
    BACKEND_PREPARE = "backend_prepare"
    COMPOSE = "compose"
    VALIDATE = "validate"


@dataclass(frozen=True)
class CompiledEnv:
    """Result of compilation, safe to inspect before ``gym.make``."""

    plan: ResolvedRunPlan
    config: Any
    snapshot: Mapping[str, Any]
    phases: tuple[CompilePhase, ...]


class EnvCompiler:
    """Compile a :class:`ResolvedRunPlan` in a fixed order.

    Hooks are deliberately small.  Isaac-specific factories can be supplied by
    applications, while tests can use plain dictionaries and fake plugins.
    """

    def __init__(
        self,
        *,
        registry: PluginRegistry | None = None,
        instantiate: Callable[[ResolvedRunPlan], Any] | None = None,
        backend_prepare: Callable[[Any, ResolvedRunPlan], Any] | None = None,
        compose: Callable[[Any, ResolvedRunPlan], Any] | None = None,
    ) -> None:
        self.registry = registry or PluginRegistry()
        self._instantiate = instantiate or (lambda plan: dict(plan.to_parse_kwargs()))
        self._backend_prepare = backend_prepare or (lambda config, _plan: config)
        self._compose = compose or (lambda config, _plan: config)

    def compile(self, plan: ResolvedRunPlan) -> CompiledEnv:
        validate_run_plan(plan)
        config = self._instantiate(plan)
        config = self._backend_prepare(config, plan)
        config = self._compose(config, plan)
        self._validate_config(config, plan)
        return CompiledEnv(
            plan=plan,
            config=config,
            snapshot=plan.snapshot(),
            phases=tuple(CompilePhase),
        )

    @staticmethod
    def _validate_config(config: Any, plan: ResolvedRunPlan) -> None:
        """Apply structural checks to dict-like configs without importing Isaac."""

        if config is None:
            raise ValueError("environment compiler produced no configuration")
        if isinstance(config, Mapping):
            declared_backend = config.get("physics_backend")
            requested_backend = plan.launch.physics_backend
            if declared_backend and requested_backend and str(declared_backend).lower() != requested_backend:
                raise ValueError(
                    "compiled config backend does not match run plan: "
                    f"{declared_backend!r} != {requested_backend!r}"
                )
