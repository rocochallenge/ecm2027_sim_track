"""Composition root for concrete physics backend capabilities."""

from typing import Any

from .common import BackendKind
from .contract import BackendLaunchProfile, BackendRuntime

def make_asset_compiler(kind: BackendKind | str):
    """Create only the selected backend asset compiler."""
    resolved = BackendKind.parse(kind)
    if resolved is BackendKind.NEWTON:
        from robofinals.core.backends.newton.assets import NewtonAssetCompiler
        return NewtonAssetCompiler()
    from robofinals.core.backends.physx.assets import PhysXAssetCompiler
    return PhysXAssetCompiler()


def make_backend_runtime(kind: BackendKind | str, *, env: Any = None) -> BackendRuntime:
    """Build a complete capability bundle for one backend."""
    resolved = BackendKind.parse(kind)
    if env is None:
        env = _EmptyEnv()
    if resolved is BackendKind.NEWTON:
        from robofinals.core.backends.newton.assets import NewtonAssetCompiler
        from robofinals.core.backends.newton.contacts import NewtonContactProvider
        from robofinals.core.backends.newton.recording import NewtonRecordingPolicy
        from robofinals.core.backends.newton.rendering import NewtonRenderer
        from robofinals.core.backends.newton.simulation import NewtonSimulationCompiler
        from robofinals.core.backends.newton.state import NewtonStateSynchronizer
        assets, contacts = NewtonAssetCompiler(), NewtonContactProvider(env)
        recording, renderer = NewtonRecordingPolicy(), NewtonRenderer()
        simulation, state = NewtonSimulationCompiler(), NewtonStateSynchronizer()
    else:
        from robofinals.core.backends.physx.assets import PhysXAssetCompiler
        from robofinals.core.backends.physx.contacts import PhysXContactProvider
        from robofinals.core.backends.physx.recording import PhysXRecordingPolicy
        from robofinals.core.backends.physx.rendering import PhysXRenderer
        from robofinals.core.backends.physx.simulation import PhysXSimulationCompiler
        from robofinals.core.backends.physx.state import PhysXStateSynchronizer
        assets, contacts = PhysXAssetCompiler(), PhysXContactProvider(env)
        recording, renderer = PhysXRecordingPolicy(), PhysXRenderer()
        simulation, state = PhysXSimulationCompiler(), PhysXStateSynchronizer()
    return BackendRuntime(
        kind=resolved,
        launch=BackendLaunchProfile(backend=resolved, kit_overrides=simulation.kit_launch_overrides(False)),
        simulation=simulation, assets=assets, contacts=contacts, state=state, renderer=renderer, recording=recording,
    )


def attach_backend_runtime(env: Any, kind: BackendKind | str) -> BackendRuntime:
    """Attach an isolated capability bundle to a live environment.

    ``backend_runtime`` is the sole capability entry point on the environment.
    """

    runtime = make_backend_runtime(kind, env=env)
    setattr(env, "backend_runtime", runtime)
    return runtime


class _EmptyEnv:
    scene = None
