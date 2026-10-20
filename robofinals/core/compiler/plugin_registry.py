"""Explicit plugin registry used by the configuration compiler.

The legacy application discovers modules through import side effects.  This registry
keeps discovery deterministic while still allowing callers to register existing
IsaacLab/Gym factories during migration.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable


class PluginRegistrationError(ValueError):
    """Raised for duplicate or invalid plugin registrations."""


@dataclass(frozen=True)
class Plugin:
    identifier: str
    factory: Callable[..., Any]
    capabilities: frozenset[str] = frozenset()


class PluginRegistry:
    """Small explicit registry with no import-time global state."""

    def __init__(self, plugins: Iterable[Plugin] = ()) -> None:
        self._plugins: dict[str, Plugin] = {}
        for plugin in plugins:
            self.register(plugin.identifier, plugin.factory, plugin.capabilities)

    def register(
        self,
        identifier: str,
        factory: Callable[..., Any],
        capabilities: Iterable[str] = (),
        *,
        replace: bool = False,
    ) -> Plugin:
        if not isinstance(identifier, str) or not identifier.strip():
            raise PluginRegistrationError("plugin identifier must be a non-empty string")
        if not callable(factory):
            raise PluginRegistrationError(f"plugin {identifier!r} factory must be callable")
        if identifier in self._plugins and not replace:
            raise PluginRegistrationError(f"plugin {identifier!r} is already registered")
        plugin = Plugin(identifier, factory, frozenset(str(item) for item in capabilities))
        self._plugins[identifier] = plugin
        return plugin

    def get(self, identifier: str) -> Plugin:
        try:
            return self._plugins[identifier]
        except KeyError as exc:
            raise PluginRegistrationError(f"plugin {identifier!r} is not registered") from exc

    def maybe_get(self, identifier: str) -> Plugin | None:
        return self._plugins.get(identifier)

    def identifiers(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))

    def instantiate(self, identifier: str, *args: Any, **kwargs: Any) -> Any:
        return self.get(identifier).factory(*args, **kwargs)
