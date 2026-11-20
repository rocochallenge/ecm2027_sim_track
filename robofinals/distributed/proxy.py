"""Restricted policy-facing IPC protocol."""

from multiprocessing.managers import BaseManager, BaseProxy


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
        if getattr(self._runtime, "_eval_config_locked", False):
            seed = None
        result = _tensors_to_cpu(self._runtime.reset(seed=seed))
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            return result[0], {}
        return result, {}

    def step(self, action):
        # Keep the policy-facing contract friendly to NumPy/list policies while
        # normalizing before the simulator calls ``action.to(device)``.
        try:
            import torch
            if not hasattr(action, "to"):
                action = torch.as_tensor(action, dtype=torch.float32)
        except ImportError:
            pass
        return _tensors_to_cpu(self._runtime.step(_tensors_to_cpu(action)))

    def get_metadata(self):
        env = self._runtime._env
        if env is None:
            raise RuntimeError("environment is not attached")
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        metadata = {
            "actions_dim": int(env.action_space.shape[-1]),
            "decimation": int(unwrapped.cfg.decimation),
            "num_envs": int(getattr(env, "num_envs", 1)),
        }
        manager = getattr(unwrapped, "action_manager", None)
        if (metadata["actions_dim"] == 19 and manager is not None
                and "base_velocity_action" in manager.active_terms
                and callable(getattr(manager.get_term("base_velocity_action"), "adapt_full_action", None))):
            metadata["accepted_action_dims"] = [19, 20]
            metadata["action_formats"] = {
                "19": "arms_grippers_16+body_velocity_vx_vy_wz",
                "20": "arms_grippers_16+legacy_wheels_w1_w2_w3_w4",
            }
        return metadata

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
        self.accepted_action_dims = None
        self.action_formats = None
        self.decimation = None
        self.num_envs = 1

    def attach(self, env_cfg):
        metadata = self._svc.attach(env_cfg)
        self.actions_dim = metadata["actions_dim"]
        self.accepted_action_dims = metadata.get("accepted_action_dims", [self.actions_dim])
        self.action_formats = metadata.get("action_formats", {})
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
        service = self._svc
        if service is None:
            return
        try:
            return service.close()
        finally:
            if isinstance(service, BaseProxy):
                # close() shuts down this server. Do not reconnect for decref
                # during proxy garbage collection: the shutdown may still be
                # completing and that authentication exchange can block.
                service._close.cancel()
                connection = getattr(service._tls, "connection", None)
                if connection is not None:
                    connection.close()
                    del service._tls.connection
            self._svc = None
