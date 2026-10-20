"""Reset coordination for full and partial vectorized environments."""

from __future__ import annotations

from typing import Any, Iterable

from .lifecycle import EpisodeState, LifecycleError


class ResetCoordinator:
    def __init__(self, lifecycle: Any, synchronizer: Any = None) -> None:
        self.lifecycle = lifecycle
        self.synchronizer = synchronizer

    def reset(self, env_ids: Iterable[int] | None = None, *, seed: int | None = None) -> Any:
        self._ensure_open()
        ids = self.lifecycle.machine.episodes.ids(env_ids)
        self.lifecycle.machine.episodes.mark(EpisodeState.RESET_PENDING, ids)
        result = self.lifecycle.env.reset(seed=seed, env_ids=_backend_env_ids(env_ids, ids))
        if self.synchronizer is not None:
            # Isaac reset updates backend state buffers first; the synchronizer
            # then publishes those transforms to USD/Fabric for interactive
            # renderers (especially Newton).
            self.synchronizer.sync_after_state_reset(self.lifecycle.env)
        self.lifecycle.machine.episodes.mark(EpisodeState.ACTIVE, ids)
        return result

    def reset_to(
        self,
        state: Any,
        env_ids: Iterable[int] | None = None,
        *,
        seed: int | None = None,
        is_relative: bool = False,
    ) -> Any:
        self._ensure_open()
        ids = self.lifecycle.machine.episodes.ids(env_ids)
        self.lifecycle.machine.episodes.mark(EpisodeState.RESET_PENDING, ids)
        reset_to = getattr(self.lifecycle.env, "reset_to", None)
        if not callable(reset_to):
            raise LifecycleError("environment does not implement reset_to")
        result = reset_to(
            state,
            env_ids=_backend_env_ids(env_ids, ids),
            seed=seed,
            is_relative=is_relative,
        )
        if self.synchronizer is not None:
            self.synchronizer.sync_after_state_reset(self.lifecycle.env)
        self.lifecycle.machine.episodes.mark(EpisodeState.ACTIVE, ids)
        return result

    def mark_terminated(self, env_ids: Iterable[int] | None = None) -> None:
        self.lifecycle.machine.episodes.mark(EpisodeState.TERMINATED, env_ids)

    def mark_truncated(self, env_ids: Iterable[int] | None = None) -> None:
        self.lifecycle.machine.episodes.mark(EpisodeState.TRUNCATED, env_ids)

    def _ensure_open(self) -> None:
        if self.lifecycle.machine.host_state in {"closing", "closed"}:
            raise LifecycleError("cannot reset a closing or closed environment")


def _backend_env_ids(original: Iterable[int] | None, normalized: tuple[int, ...]) -> Any:
    """Preserve tensor-like ids required by vectorized backend implementations.

    The lifecycle state machine deliberately normalizes ids to a tuple for
    bookkeeping. Isaac backends, however, may require the caller's tensor type
    (for example Newton's reset path indexes CUDA buffers with ``.shape`` and
    ``.device``). Plain Python iterables keep the stable tuple contract.
    """
    if original is None:
        return None
    if hasattr(original, "shape") and hasattr(original, "device"):
        return original
    return normalized
