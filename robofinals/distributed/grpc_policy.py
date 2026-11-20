"""Policy-only protobuf transport; simulator runs on the calling thread."""
from concurrent import futures
from dataclasses import dataclass, field
import queue
import threading
import time

import grpc
import numpy as np

from . import policy_v1_pb2 as pb
from .chamber_schema import JOINT_DEFAULT_POS, JOINT_NAMES, OBSERVATION_SCHEMA

_PATH = "/ecm27.policy.v1.Policy/"
_MAX_BYTES = 64 * 1024 * 1024
# Worst case is an observation and a terminal observation for every slot.
_MAX_ENVS = 27


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype not in (np.dtype("uint8"), np.dtype("float32")):
        raise ValueError("unsupported observation dtype")
    return np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))


def _observation(observation, expected_count=None):
    policy = observation.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("policy observation group missing")
    arrays = {}
    count = None
    for name, dtype, trailing_shape in OBSERVATION_SCHEMA:
        if name not in policy:
            raise ValueError(f"required observation missing: {name}")
        arr = _array(policy[name])
        if count is None:
            count = arr.shape[0] if arr.ndim else 0
        if count < 1 or arr.shape != (count, *trailing_shape) or str(arr.dtype) != dtype:
            raise ValueError(f"invalid observation schema: {name}, got {arr.shape}/{arr.dtype}")
        arrays[name] = arr
    if expected_count is not None and count != expected_count:
        raise ValueError("observation batch size does not match configured environment count")
    return [pb.Observation(**{name: np.ascontiguousarray(array[i]).tobytes()
                              for name, array in arrays.items()}) for i in range(count)]


def _flags(value, count):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.bool_).reshape(-1)
    if arr.size != count:
        raise ValueError("termination flag length mismatch")
    return arr.tolist()


class InvalidAction(ValueError):
    """A policy request violates the action contract."""


class PendingWallTimeout(RuntimeError):
    """A timed-out episode needs its reset observation consumed first."""


@dataclass
class _Request:
    method: str
    payload: object
    context: object
    done: threading.Event = field(default_factory=threading.Event)
    result: object = None
    error: object = None


class GrpcPolicyServer:
    def __init__(self, env_initializer, address, record_dir=None, record_metadata=None,
                 wall_timeout_seconds=300*60):
        self._env_initializer = env_initializer
        self._env = None
        self.address = address
        self.record_dir = record_dir
        self.record_metadata = record_metadata or {}
        self.wall_timeout_seconds = float(wall_timeout_seconds)
        if not np.isfinite(self.wall_timeout_seconds) or self.wall_timeout_seconds <= 0:
            raise ValueError("wall timeout must be finite and positive")
        self._recorder = None
        self._stop = threading.Event()
        self._requests = queue.Queue(maxsize=1)
        self._inflight = threading.Lock()
        self._grpc = None
        self._signal_handlers = {}

    def _call(self, method, request, context):
        if not self._inflight.acquire(blocking=False):
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "another request is in progress")
        try:
            item = _Request(method, request, context)
            self._requests.put_nowait(item)
            while not item.done.wait(0.1):
                if not context.is_active() or self._stop.is_set():
                    context.abort(grpc.StatusCode.CANCELLED, "request cancelled")
            if item.error:
                code, message = item.error
                context.abort(code, message)
            return item.result
        finally:
            self._inflight.release()

    def _handle(self, item):
        if not item.context.is_active():
            return
        if item.method == "GetObservation":
            item.result = self._transition
            consumed = [i for i, pending in enumerate(self._pending_wall_timeout) if pending]
            if consumed:
                if self._recorder is not None:
                    self._recorder.start_slots(self._last_raw_observation, consumed)
                now = time.monotonic()
                for i in consumed:
                    self._pending_wall_timeout[i] = False
                    self._pending_final[i] = None
                    self._episode_started[i] = now
        else:
            if any(self._pending_wall_timeout):
                raise PendingWallTimeout("episode timed out; call GetObservation before Step")
            rows = [list(row.values) for row in item.payload.actions]
            if len(rows) != self._n or not rows or len(rows[0]) not in (19, 20):
                raise InvalidAction("send one 19-dimensional or 20-dimensional action per environment")
            if any(len(row) != len(rows[0]) for row in rows):
                raise InvalidAction("all actions in a batch must use the same dimension")
            action = np.asarray(rows, dtype=np.float32)
            if not np.isfinite(action).all():
                raise InvalidAction("action contains non-finite values")
            # Numerical validity only; invalid requests do not advance or score the simulation.
            if np.any(np.abs(action[:, (0, 1, 2, 7, 8, 9)]) > 10.0):
                raise InvalidAction("arm target coordinates must be within +/-10 metres")
            if action.shape[1] == 19:
                base = action[:, 16:19].astype(np.float64)
            else:
                wheels = action[:, 16:20].astype(np.float64)
                if np.any(np.abs(wheels) > 30.0):
                    raise InvalidAction("legacy wheel action values must be within +/-30")
                w1, w2, w3, w4 = wheels.T
                base = np.stack(((w1+w2+w3+w4)/4, (-w1+w2+w3-w4)/4,
                                 (-w1+w2-w3+w4)/4), axis=1)
            if np.any(np.abs(base) > 10.0):
                raise InvalidAction("base velocity components must be within +/-10")
            applied_action = action.copy()
            for part in (slice(3, 7), slice(10, 14)):
                norms = np.linalg.norm(action[:, part].astype(np.float64), axis=1)
                if np.any(norms < 1e-6):
                    raise InvalidAction("arm target quaternion norm must be at least 1e-6")
                applied_action[:, part] = action[:, part] / norms[:, None]
            import torch
            if self._recorder is not None:
                self._recorder.before_step(action, applied_action=applied_action)
            try:
                observation, _reward, terminated, truncated, _info = self._env.step(torch.from_numpy(applied_action))
            except Exception as exc:
                if self._recorder is not None:
                    self._recorder.step_error(exc)
                raise
            if self._recorder is not None:
                self._recorder.after_step(observation)
            self._last_raw_observation = observation
            ends = _flags(terminated, self._n); limits = _flags(truncated, self._n)
            for i in range(self._n):
                if ends[i] or limits[i]:
                    self._episode_started[i] = time.monotonic()
            final_observation = []
            if any(ends) or any(limits):
                if not isinstance(_info, dict) or "final_obs" not in _info:
                    raise RuntimeError("environment did not provide terminal observations")
                final_observation = _observation(_info["final_obs"], self._n)
            self._transition = pb.Transition(observations=_observation(observation, self._n), terminated=ends,
                                             truncated=limits, final_observations=final_observation)
            item.result = self._transition

    def _expire_overdue(self):
        now = time.monotonic()
        expired = [i for i in range(self._n) if not self._pending_wall_timeout[i]
                   and now - self._episode_started[i] >= self.wall_timeout_seconds]
        if not expired:
            return
        import torch
        raw = self._env.unwrapped
        for i in expired:
            self._pending_final[i] = pb.Observation.FromString(
                self._transition.observations[i].SerializeToString())
        # Preserve the current camera frame and scoring state before reset.
        raw.extras["final_obs"] = raw.observation_manager.compute(update_history=False)
        raw._forced_episode_end_reason = "wall_clock_timeout"
        try:
            ids = torch.tensor(expired, dtype=torch.long, device=raw.device)
            observation, _ = self._env.reset(env_ids=ids)
        finally:
            raw._forced_episode_end_reason = None
        self._last_raw_observation = observation
        for i in expired:
            self._pending_wall_timeout[i] = True
        current = _observation(observation, self._n)
        finals = [self._pending_final[i] if self._pending_wall_timeout[i] else current[i]
                  for i in range(self._n)]
        self._transition = pb.Transition(observations=current, terminated=[False]*self._n,
            truncated=self._pending_wall_timeout, final_observations=finals)

    def serve(self):
        self._env = self._env_initializer(None)
        obs, _ = self._env.reset()
        self._n = int(self._env.num_envs)
        self._dt = float(self._env.unwrapped.step_dt)
        if self._n < 1 or self._n > _MAX_ENVS:
            raise ValueError("gRPC contract supports between 1 and 27 environment slots")
        if int(self._env.action_space.shape[-1]) != 19:
            raise ValueError("evaluation environment must use the 19-dimensional Galbot-Mobile action")
        if abs(self._dt - 0.02) > 1e-8:
            raise ValueError("evaluation control step must be 0.02 seconds")
        robot = self._env.unwrapped.scene["robot"]
        actual_joints = tuple(robot.data.joint_names)
        if actual_joints != JOINT_NAMES:
            raise ValueError("evaluation robot joint order does not match the policy contract")
        from robofinals.utils.isaac_data_compat import as_torch
        positions = as_torch(robot.data.default_joint_pos).detach().cpu().numpy()
        velocities = as_torch(robot.data.default_joint_vel).detach().cpu().numpy()
        if positions.shape != (self._n, 77) or velocities.shape != (self._n, 77):
            raise ValueError("default joint states do not match the policy contract")
        if not np.allclose(positions, np.asarray(JOINT_DEFAULT_POS)[None, :], rtol=0, atol=1e-6):
            raise ValueError("default joint positions do not match the policy contract")
        if not np.allclose(velocities, 0.0, rtol=0, atol=1e-6):
            raise ValueError("default joint velocities do not match the policy contract")
        self._episode_started = [time.monotonic()] * self._n
        self._pending_wall_timeout = [False] * self._n
        self._pending_final = [None] * self._n
        self._last_raw_observation = obs
        if self.record_dir is not None:
            from robofinals.applications.eval.episode_records import EpisodeRecords
            self._recorder = EpisodeRecords(self.record_dir, self._env, self.record_metadata)
            self._recorder.start(obs)
            print(f"[grpc_policy] evaluation run directory: {self._recorder.root}", flush=True)
        self._transition = pb.Transition(observations=_observation(obs, self._n), terminated=[False]*self._n,
                                         truncated=[False]*self._n)
        self._grpc = grpc.server(futures.ThreadPoolExecutor(max_workers=2), options=(
            ("grpc.max_receive_message_length", _MAX_BYTES), ("grpc.max_send_message_length", _MAX_BYTES)))
        handlers = {}
        for method, request_type, response_type in (("GetObservation", pb.Empty, pb.Transition),
                                                     ("Step", pb.ActionBatch, pb.Transition)):
            handlers[method] = grpc.unary_unary_rpc_method_handler(
                lambda request, context, method=method: self._call(method, request, context),
                request_deserializer=request_type.FromString, response_serializer=response_type.SerializeToString)
        self._grpc.add_generic_rpc_handlers((grpc.method_handlers_generic_handler("ecm27.policy.v1.Policy", handlers),))
        host, port = self.address
        if not self._grpc.add_insecure_port(f"{host}:{port}"):
            raise RuntimeError("gRPC listener could not bind")
        self._grpc.start()
        # AppLauncher can install its own signal handlers, so register after initialization.
        if threading.current_thread() is threading.main_thread():
            import signal
            for signum in (signal.SIGINT, signal.SIGTERM):
                self._signal_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, lambda *_: self._stop.set())
        self._episode_started = [time.monotonic()] * self._n
        print(f"[grpc_policy] listening on {host}:{port}", flush=True)
        while not self._stop.is_set():
            self._expire_overdue()
            try:
                item = self._requests.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._expire_overdue()
                self._handle(item)
            except PendingWallTimeout as exc:
                item.error = (grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            except InvalidAction as exc:
                item.error = (grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            except Exception:
                import traceback
                traceback.print_exc()
                item.error = (grpc.StatusCode.INTERNAL, "simulator operation failed")
            finally:
                item.done.set()

    def close(self):
        self._stop.set()
        if self._signal_handlers and threading.current_thread() is threading.main_thread():
            import signal
            for signum, handler in self._signal_handlers.items():
                signal.signal(signum, handler)
            self._signal_handlers.clear()
        if self._grpc is not None:
            self._grpc.stop(grace=1).wait(timeout=3)
            self._grpc = None
        if self._recorder is not None:
            self._recorder.close()
            self._recorder = None
        if self._env is not None:
            self._env.close()
            self._env = None


class PolicyClient:
    """Minimal stub. Retry only requests known not to have advanced simulation."""
    MAX_RETRIES = 3

    def __init__(self, channel):
        self._get_observation = channel.unary_unary(_PATH+"GetObservation",
            request_serializer=pb.Empty.SerializeToString, response_deserializer=pb.Transition.FromString)
        self._step = channel.unary_unary(_PATH+"Step",
            request_serializer=pb.ActionBatch.SerializeToString, response_deserializer=pb.Transition.FromString)

    @staticmethod
    def _invoke(call, request, *, timeout, step):
        for attempt in range(PolicyClient.MAX_RETRIES + 1):
            try:
                return call(request, timeout=timeout)
            except grpc.RpcError as exc:
                if step:
                    # This exact server rejection occurs before enqueuing any action.
                    safe = (exc.code() == grpc.StatusCode.RESOURCE_EXHAUSTED and
                            exc.details() == "another request is in progress")
                else:
                    safe = exc.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED,
                                          grpc.StatusCode.RESOURCE_EXHAUSTED)
                if not safe or attempt >= PolicyClient.MAX_RETRIES:
                    raise
                time.sleep(0.1 * (2 ** attempt))

    def get_observation(self, request, *, timeout=None):
        return self._invoke(self._get_observation, request, timeout=timeout, step=False)

    def step(self, request, *, timeout=None):
        return self._invoke(self._step, request, timeout=timeout, step=True)
