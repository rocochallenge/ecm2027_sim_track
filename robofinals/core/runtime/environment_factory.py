"""Single environment construction entry point for applications."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping
from dataclasses import replace

from robofinals.core.backends import attach_backend_runtime
from robofinals.core.compiler import EnvCompiler
from robofinals.core.compiler.env_compiler import CompiledEnv
from robofinals.core.runtime import EnvironmentLifecycle

from .run_plan import ConfigResolver, ResolvedRunPlan


class EnvironmentFactory:
    """Compile a plan and construct the shared runtime boundary.

    Parser and environment-maker hooks keep Isaac imports at the composition
    root while the returned value is always an :class:`EnvironmentLifecycle`.
    """

    def __init__(
        self,
        parser: Callable[..., Any] | None = None,
        make_env: Callable[[CompiledEnv], Any] | None = None,
    ):
        self._parser = parser
        self._make_env_hook = make_env
        self._compiler = EnvCompiler()

    def compile(self, plan: ResolvedRunPlan, snapshot_path: str | Path | None = None) -> CompiledEnv:
        compiled = self._compiler.compile(plan)
        if snapshot_path is not None:
            plan.write_snapshot(snapshot_path)
        return compiled

    def create(
        self,
        plan: ResolvedRunPlan | CompiledEnv,
        snapshot_path: str | Path | None = None,
    ) -> EnvironmentLifecycle:
        compiled = plan if isinstance(plan, CompiledEnv) else self.compile(plan, snapshot_path=snapshot_path)
        parser = self._parser
        if parser is None:
            from robofinals.utils.env import parse_env_cfg

            parser = parse_env_cfg
        kwargs = compiled.plan.to_parse_kwargs()
        from robofinals.utils.env import str_to_execute_mode

        kwargs["execute_mode"] = str_to_execute_mode(compiled.plan.application.execute_mode)
        compiled = replace(compiled, config=parser(**kwargs))

        return self._lifecycle_from_compiled(compiled)

    def _lifecycle_from_compiled(self, compiled: CompiledEnv) -> EnvironmentLifecycle:
        env = self._make_env(compiled)
        backend = attach_backend_runtime(env, compiled.plan.launch.resolved_backend)
        return EnvironmentLifecycle(env, state_synchronizer=backend.state)

    def create_from_kwargs(
        self, values: Mapping[str, Any] | None = None, snapshot_path: str | Path | None = None, **kwargs: Any
    ) -> Any:
        merged = dict(values or {})
        merged.update(kwargs)
        return self.create(ConfigResolver.resolve(cli_values=merged), snapshot_path=snapshot_path)

    def compile_config_from_kwargs(
        self,
        values: Mapping[str, Any] | None = None,
        snapshot_path: str | Path | None = None,
        **kwargs: Any,
    ) -> Any:
        """Compile and parse an Isaac config without constructing an environment."""

        merged = dict(values or {})
        merged.update(kwargs)
        plan = ConfigResolver.resolve(cli_values=merged)
        compiled = self.compile(plan, snapshot_path=snapshot_path)
        from robofinals.utils.env import parse_env_cfg, str_to_execute_mode

        parse_kwargs = plan.to_parse_kwargs()
        parse_kwargs["execute_mode"] = str_to_execute_mode(plan.application.execute_mode)
        return replace(compiled, config=parse_env_cfg(**parse_kwargs)).config

    def create_from_config(
        self,
        plan: ResolvedRunPlan,
        config: Any,
        *,
        env_name: str | None = None,
    ) -> EnvironmentLifecycle:
        """Create a lifecycle from a parser-produced config.

        This is the migration bridge for Isaac entry points that need to keep
        their task-specific parser options while moving ``gym.make`` and
        runtime ownership into the factory.
        """

        compiled = self._compiler.compile(plan)
        if env_name is not None and not getattr(config, "env_name", None):
            try:
                config.env_name = env_name
            except (AttributeError, TypeError):
                pass
        compiled = replace(compiled, config=config)
        return self._lifecycle_from_compiled(compiled)

    def _make_env(self, compiled: CompiledEnv) -> Any:
        if self._make_env_hook is not None:
            return self._make_env_hook(compiled)
        return _make_isaac_env(compiled)


def _make_isaac_env(compiled: CompiledEnv) -> Any:
    """Default IsaacLab maker kept in the application composition root."""

    import gymnasium as gym

    config = compiled.config
    task_name = getattr(config, "env_name", None) or compiled.plan.extra.get("env_name")
    if not task_name:
        task_name = f"Robocasa-{compiled.plan.env.task_name}-{compiled.plan.env.robot_name}-v0"
    if task_name not in gym.envs.registry:
        gym.register(
            id=task_name,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={},
            disable_env_checker=True,
        )
    return gym.make(task_name, cfg=config).unwrapped
