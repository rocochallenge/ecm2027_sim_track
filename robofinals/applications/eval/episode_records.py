"""Per-slot evaluation artifacts. All methods are called on the simulator thread."""
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

CAMERAS = ("first_person_camera_rgb", "left_hand_camera_rgb", "right_hand_camera_rgb")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class CrashSafeVideoWriter:
    """Own FFmpeg process finalizes on pipe EOF, including parent SIGKILL."""

    def __init__(self, path, width, height, fps):
        self.path = Path(path)
        self._log = self.path.with_suffix(".encoder.log").open("wb")
        args = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-preset", "ultrafast", "-threads", "2", "-pix_fmt", "yuv420p",
                "-g", "10", "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
                "-f", "mp4", str(self.path)]
        try:
            self._process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                             stderr=self._log, start_new_session=True)
        except BaseException:
            self._log.close()
            raise

    def write(self, rgb):
        if self._process.poll() is not None:
            raise RuntimeError(f"video encoder exited: {self.path}")
        self._process.stdin.write(np.ascontiguousarray(rgb).tobytes())
        # Do not leave a partial last frame in the parent process buffer.
        self._process.stdin.flush()

    def release(self):
        try:
            if self._process.stdin and not self._process.stdin.closed:
                self._process.stdin.close()
            result = self._process.wait(timeout=60)
            if result:
                raise RuntimeError(f"video encoder failed ({result}): {self.path}")
        finally:
            self._log.close()


class EpisodeRecords:
    def __init__(self, root, env, metadata):
        self.env = env
        self.task = env.cfg.isaaclab_arena_env.task
        self.count = int(env.num_envs)
        self.dt = float(env.unwrapped.step_dt)
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:10]
        self.root = Path(root) / name
        self.root.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata)
        self.active = [None] * self.count
        self.sequence = [0] * self.count
        self.task._episode_finished_callback = self._on_finish
        self._write_json(self.root / "run.json", {"created_utc": _now(), "num_envs": self.count,
                                                   "step_seconds": self.dt, **self.metadata})
        # Independent helper notices evaluator death through pipe EOF and marks
        # unfinished manifests after the independent encoders close their files.
        self._recovery_log = (self.root / "recovery.log").open("wb")
        try:
            self._recovery = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("episode_recovery.py")), str(self.root)],
                stdin=subprocess.PIPE, stdout=self._recovery_log, stderr=self._recovery_log,
                start_new_session=True)
        except BaseException:
            self._recovery_log.close()
            raise

    @staticmethod
    def _write_json(path, data):
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)

    def _event(self, state, kind, **data):
        state["events"].write(json.dumps({"type": kind, "utc": _now(), **data}, ensure_ascii=False) + "\n")
        state["events"].flush()

    def _start(self, env_id):
        episode = self.sequence[env_id]
        self.sequence[env_id] += 1
        directory = self.root / f"env_{env_id:04d}" / f"episode_{episode:06d}"
        directory.mkdir(parents=True, exist_ok=False)
        state = {"directory": directory, "env_id": env_id, "episode_id": episode,
                 "events": (directory / "events.jsonl").open("a", encoding="utf-8"),
                 "writers": {}, "frames": {}, "points": set(), "stages": set(), "steps": 0,
                 "started_utc": _now(), "started_monotonic": time.monotonic()}
        self.active[env_id] = state
        self._write_json(directory / "manifest.json", {"status": "recording", "env_id": env_id,
            "episode_id": episode, "started_utc": state["started_utc"],
            "started_monotonic": state["started_monotonic"]})
        self._event(state, "episode_started", sim_time_s=0.0)
        return state

    def _frames(self, state, observation):
        group = observation.get("policy", {})
        for camera in CAMERAS:
            if camera not in group:
                raise RuntimeError(f"required recording camera missing: {camera}")
            frame = _numpy(group[camera][state["env_id"]])
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError(f"invalid RGB recording frame: {camera}")
            height, width = frame.shape[:2]
            if camera not in state["writers"]:
                writer = CrashSafeVideoWriter(state["directory"] / f"{camera}.mp4",
                                              width, height, 1.0/self.dt)
                state["writers"][camera] = writer
                (state["directory"] / f"{camera}.encoder.pid").write_text(str(writer._process.pid), encoding="utf-8")
                state["frames"][camera] = 0
            state["writers"][camera].write(frame)
            state["frames"][camera] += 1

    def start(self, observation):
        self.start_slots(observation, range(self.count))

    def start_slots(self, observation, env_ids):
        for i in env_ids:
            if self.active[i] is not None:
                raise RuntimeError(f"episode already active: {i}")
            state = self._start(i)
            self._frames(state, observation)

    def before_step(self, action, applied_action=None):
        for i, state in enumerate(self.active):
            if state is not None:
                state["steps"] += 1
                values = {"values": action[i].tolist()}
                if applied_action is not None:
                    values["applied_values"] = applied_action[i].tolist()
                self._event(state, "action", step=state["steps"],
                            sim_time_s=round(state["steps"]*self.dt, 6), **values)

    def step_error(self, error):
        for state in self.active:
            if state is not None:
                self._event(state, "simulator_error", error_type=type(error).__name__, message=str(error))

    def _score_events(self, state, metrics):
        sim_time = metrics["episode_sim_time_s"]
        for name, awarded in metrics["points"].items():
            if awarded and name not in state["points"]:
                state["points"].add(name)
                self._event(state, "score_point", point=name, sim_time_s=sim_time)
        for name in metrics["completed_stages"]:
            if name not in state["stages"]:
                state["stages"].add(name)
                self._event(state, "stage_completed", stage=name, sim_time_s=sim_time)

    def _current_metrics(self, i):
        points = [bool(v) for v in self.task._point_flags[i].detach().cpu().tolist()]
        stages = [bool(v) for v in self.task._stage_flags[i].detach().cpu().tolist()]
        return {"points": dict(zip(self.task.POINT_NAMES, points)),
                "completed_stages": [n for n, flag in zip(self.task._stage_names, stages) if flag],
                "episode_sim_time_s": round(int(self.task._score_steps[i].item()) * self.dt, 6)}

    def after_step(self, observation):
        for i in range(self.count):
            state = self.active[i]
            if state is None:
                state = self._start(i)
            else:
                self._score_events(state, self._current_metrics(i))
            self._frames(state, observation)

    def _on_finish(self, env_id, metrics):
        state = self.active[env_id]
        if state is None:
            return
        raw = self.env.unwrapped
        # The callback runs inside Isaac's pre-reset event, after termination computation.
        metrics = dict(metrics)
        if metrics.get("end_reason") == "wall_clock_timeout":
            terminated, truncated = False, True
        else:
            terminated = bool(raw.termination_manager.terminated[env_id])
            truncated = bool(raw.termination_manager.time_outs[env_id])
            metrics["end_reason"] = ("terminated_and_truncated" if terminated and truncated else
                                     "terminated" if terminated else "truncated" if truncated else "reset")
        metrics["terminated"] = terminated
        metrics["truncated"] = truncated
        # Capture the last available camera observation before reset mutates the scene.
        final_obs = raw.extras.get("final_obs")
        self._frames(state, final_obs if final_obs is not None else raw.observation_manager.compute())
        self._score_events(state, metrics)
        self._event(state, "episode_finished", reason=metrics["end_reason"], sim_time_s=metrics["episode_sim_time_s"])
        self._finish(state, metrics, "complete")
        self.active[env_id] = None

    def _finish(self, state, metrics, status):
        state["events"].flush(); os.fsync(state["events"].fileno()); state["events"].close()
        for writer in state["writers"].values(): writer.release()
        files = {name: {"file": name+".mp4", "frames": state["frames"][name]}
                 for name in state["frames"]}
        video_errors = []
        for name in CAMERAS:
            path = state["directory"] / (name + ".mp4")
            capture = cv2.VideoCapture(str(path))
            ok, _ = capture.read()
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            if not ok or count != state["frames"].get(name, 0):
                video_errors.append({"camera": name, "expected_frames": state["frames"].get(name, 0),
                                     "readable_frames": count})
        if video_errors:
            status = "incomplete"
        self._write_json(state["directory"] / "score.json", metrics)
        self._write_json(state["directory"] / "manifest.json", {"status": status, "env_id": state["env_id"],
            "episode_id": state["episode_id"], "started_utc": state["started_utc"], "ended_utc": _now(),
            "step_seconds": self.dt, "recorded_steps": state["steps"], "events": "events.jsonl",
            "score": "score.json", "videos": files, "video_errors": video_errors})
        if video_errors:
            print(f"[episode_records] incomplete video: {state['directory']}: {video_errors}", flush=True)

    def close(self):
        for i, state in enumerate(self.active):
            if state is None: continue
            metrics = self._current_metrics(i)
            metrics.update({"process_score": int(self.task._process_score[i].item()), "end_reason": "service_shutdown"})
            self._score_events(state, metrics)
            self._event(state, "episode_interrupted", reason="service_shutdown")
            self._finish(state, metrics, "interrupted")
            self.active[i] = None
        self.task._episode_finished_callback = None
        self._recovery.stdin.close()
        try:
            self._recovery.wait(timeout=30)
        finally:
            self._recovery_log.close()
