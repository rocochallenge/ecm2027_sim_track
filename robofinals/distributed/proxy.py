"""Restricted policy-facing IPC protocol."""

from multiprocessing.managers import BaseManager


def _tensors_to_cpu(obj):
    """Recursively move tensors to CPU before Manager serialization."""
    try:
        import torch
    except ImportError:
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: _tensors_to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        values = [_tensors_to_cpu(value) for value in obj]
        return type(obj)(values)
    return obj


# Deliberately no generic call/getattr/setattr path access.
EXPOSED_METHODS = ("attach", "reset", "step", "get_metadata", "close")


class EnvService:
    """The only service object visible to the policy process."""

    def __init__(self, runtime):
        self._runtime = runtime

    def attach(self, env_cfg):
        self._runtime.attach(env_cfg)
        return self.get_metadata()

    def reset(self, seed=None):
        result = _tensors_to_cpu(self._runtime.reset(seed=seed))
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            return result[0], {}
        return result, {}

    def step(self, action):
        return _tensors_to_cpu(self._runtime.step(_tensors_to_cpu(action)))

    def get_metadata(self):
        env = self._runtime._env
        if env is None:
            raise RuntimeError("environment is not attached")
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        return {
            "actions_dim": int(env.action_space.shape[-1]),
            "decimation": int(unwrapped.cfg.decimation),
            "num_envs": int(getattr(env, "num_envs", 1)),
        }

    def close(self):
        return self._runtime.close()


class EnvManager(BaseManager):
    def register_for_server(self, runtime):
        self.register("EnvService", lambda: EnvService(runtime), exposed=EXPOSED_METHODS)

    def register_for_client(self):
        self.register("EnvService", exposed=EXPOSED_METHODS)


class RemoteEnv:
    """Client implementing only the public policy/environment contract."""

    @classmethod
    def make(cls, address, authkey=b"lightwheel"):
        import multiprocessing

        multiprocessing.current_process().authkey = authkey
        manager = EnvManager(address=address, authkey=authkey)
        manager.connect()
        manager.register_for_client()
        return cls(manager.EnvService())

    def __init__(self, service):
        self._svc = service
        self.actions_dim = None
        self.decimation = None
        self.num_envs = 1

    def attach(self, env_cfg):
        metadata = self._svc.attach(env_cfg)
        self.actions_dim = metadata["actions_dim"]
        self.decimation = metadata["decimation"]
        self.num_envs = metadata["num_envs"]
        return metadata

    def reset(self, seed=None):
        result = self._svc.reset(seed)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}

    def step(self, action):
        result = self._svc.step(_tensors_to_cpu(action))
        if not isinstance(result, (tuple, list)) or len(result) < 5:
            raise RuntimeError("environment service returned an invalid Gym transition")
        observation, _reward, terminated, truncated, _info = result[:5]
        # Reward, info, success, and task internals stay in the simulator side.
        return observation, terminated, truncated

    def close(self):
        return self._svc.close()
