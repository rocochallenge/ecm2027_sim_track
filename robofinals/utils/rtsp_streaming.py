"""
RTSP per-env streaming helper for env_server.py.

Each simulated environment gets one FFmpeg process pushed frames from
the Isaac Lab TiledCamera tensor via an omni.kit.app update subscription.

Requirements:
  - ffmpeg on PATH.
  - mediamtx on PATH, OR an external RTSP broker already running on base_port.
    mediamtx respects the MTX_RTSPADDRESS env var to select a non-default port.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Optional

import carb
import numpy as np

_workers: list[_StreamWorker] = []
_mediamtx_proc: Optional[subprocess.Popen] = None
_mediamtx_config_path: Optional[str] = None
_update_sub = None
_in_frame_callback: bool = False  # reentrance guard
_logged_first_frame: bool = False


def _pipe_process_log(prefix: str, pipe) -> None:
    if pipe is None:
        return
    try:
        for raw_line in iter(pipe.readline, b""):
            line = raw_line.decode(errors="replace").strip()
            if line:
                print(f"{prefix} {line}", flush=True)
    except Exception as exc:
        print(f"{prefix} log reader stopped: {exc}", flush=True)


class _StreamWorker:
    def __init__(self, env_idx: int, width: int, height: int, rtsp_url: str, bitrate: str = "4M"):
        self.env_idx = env_idx
        self.width = width
        self.height = height
        self.url = rtsp_url
        self.bitrate = bitrate
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._frames_written = 0
        self._start()

    def _start(self) -> None:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", "30",
            "-i", "pipe:0",
            "-an",
            "-vf", "format=yuv420p",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-profile:v", "baseline",
            "-level", "3.1",
            "-g", "30",
            "-keyint_min", "30",
            "-sc_threshold", "0",
            "-x264-params", "repeat-headers=1",
            "-b:v", self.bitrate, "-maxrate", self.bitrate, "-bufsize", self.bitrate,
            "-f", "rtsp", "-rtsp_transport", "tcp",
            self.url,
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        threading.Thread(
            target=_pipe_process_log,
            args=(f"[rtsp_streaming][ffmpeg env_{self.env_idx}]", self._proc.stderr),
            daemon=True,
        ).start()

    def write_frame(self, frame: np.ndarray) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        if proc.poll() is not None:
            print(
                f"[rtsp_streaming] ffmpeg env_{self.env_idx} exited with code {proc.returncode}",
                flush=True,
            )
            self._proc = None
            return
        try:
            with self._lock:
                proc.stdin.write(frame.tobytes())
                proc.stdin.flush()
                self._frames_written += 1
                if self._frames_written == 1:
                    print(f"[rtsp_streaming] env_{self.env_idx} wrote first frame", flush=True)
        except (BrokenPipeError, OSError):
            print(f"[rtsp_streaming] ffmpeg env_{self.env_idx} pipe closed", flush=True)
            self._proc = None

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def setup_rtsp_streams(
    env,
    *,
    camera_name: str = "first_person_camera",
    base_port: int = 18554,
    width: int = 640,
    height: int = 360,
    mount_prefix: str = "/env",
    advertise_host: Optional[str] = None,
    max_streams: Optional[int] = None,
    bitrate: str = "4M",
) -> list[str]:
    """Start mediamtx (if available) and one FFmpeg worker per env camera.

    Returns the list of advertised RTSP URLs.
    """
    global _workers, _mediamtx_proc, _update_sub

    teardown_rtsp_streams()

    _mediamtx_proc = _try_start_mediamtx(base_port)

    cameras = _find_cameras(env, camera_name)
    if not cameras:
        available = _available_sensor_names(env)
        print(
            f"[rtsp_streaming] WARNING: no sensor named '{camera_name}' found in scene; "
            f"no streams started. available sensors={available}"
        )
        return []

    host = advertise_host or "127.0.0.1"
    num = min(len(cameras), max_streams) if max_streams is not None else len(cameras)

    urls: list[str] = []
    for i in range(num):
        mount = f"{mount_prefix}_{i}"
        url = f"rtsp://{host}:{base_port}{mount}"
        worker = _StreamWorker(i, width, height, url, bitrate=bitrate)
        _workers.append(worker)
        urls.append(url)
        print(f"[rtsp_streaming] env_{i} stream → {url}")

    # Cache sensor at setup time to avoid per-frame scene lookup triggering re-entry.
    sensor = cameras[0]

    import omni.kit.app
    _update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
        lambda _e, _s=sensor, _w=width, _h=height: _on_frame(_s, _w, _h),
        name="rtsp_streaming_frame_push",
    )

    return urls


def _find_cameras(env, camera_name: str) -> list:
    """Return list of matching camera sensors from env.scene."""
    scene = getattr(env, "scene", None)
    if scene is None:
        return []
    try:
        sensor = scene[camera_name]
        return [sensor]
    except (KeyError, TypeError):
        pass
    sensors = getattr(scene, "_sensors", {})
    matched = [v for k, v in sensors.items() if k == camera_name]
    return matched


def _available_sensor_names(env) -> list[str]:
    scene = getattr(env, "scene", None)
    if scene is None:
        return []
    sensors = getattr(scene, "_sensors", {})
    try:
        return sorted(str(k) for k in sensors.keys())
    except Exception:
        return []


def _on_frame(sensor, width: int, height: int) -> None:
    global _workers, _in_frame_callback, _logged_first_frame
    if _in_frame_callback or not _workers:
        return
    _in_frame_callback = True
    try:
        rgb = sensor.data.output.get("rgb", None)
        if rgb is None:
            return
        rgb_np: np.ndarray = (
            rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
        ).astype(np.uint8)
        if not _logged_first_frame:
            print(
                f"[rtsp_streaming] first camera tensor shape={rgb_np.shape}, dtype={rgb_np.dtype}",
                flush=True,
            )
            _logged_first_frame = True

        for worker in _workers:
            idx = worker.env_idx
            if idx >= rgb_np.shape[0]:
                continue
            frame = rgb_np[idx]  # H x W x C
            fh, fw = frame.shape[:2]
            if fh != height or fw != width:
                import cv2
                frame = cv2.resize(frame, (width, height))
            worker.write_frame(frame)
    except Exception as exc:
        carb.log_warn(f"[rtsp_streaming] frame error: {exc}")
    finally:
        _in_frame_callback = False


def teardown_rtsp_streams() -> None:
    """Stop all FFmpeg workers, the update subscription, and mediamtx (if managed)."""
    global _workers, _mediamtx_proc, _mediamtx_config_path, _update_sub

    _update_sub = None

    for w in list(_workers):
        w.stop()
    _workers.clear()

    if _mediamtx_proc is not None:
        _mediamtx_proc.terminate()
        try:
            _mediamtx_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _mediamtx_proc.kill()
        _mediamtx_proc = None

    if _mediamtx_config_path is not None:
        try:
            os.unlink(_mediamtx_config_path)
        except OSError:
            pass
        _mediamtx_config_path = None


def _try_start_mediamtx(port: int) -> Optional[subprocess.Popen]:
    global _mediamtx_config_path

    if shutil.which("mediamtx") is None:
        print(
            f"[rtsp_streaming] mediamtx not in PATH; "
            f"assuming external RTSP broker on port {port}."
        )
        return None
    config_file = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="robofinals_mediamtx_",
        suffix=".yml",
        delete=False,
    )
    with config_file:
        config_file.write(
            f"""rtspAddress: :{port}
paths:
  all_others:
    source: publisher
"""
        )
    _mediamtx_config_path = config_file.name

    try:
        proc = subprocess.Popen(
            ["mediamtx", _mediamtx_config_path],
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        threading.Thread(
            target=_pipe_process_log,
            args=("[rtsp_streaming][mediamtx]", proc.stdout),
            daemon=True,
        ).start()
        threading.Thread(
            target=_pipe_process_log,
            args=("[rtsp_streaming][mediamtx]", proc.stderr),
            daemon=True,
        ).start()
        time.sleep(0.5)
        if proc.poll() is not None:
            print(f"[rtsp_streaming] mediamtx exited with code {proc.returncode}", flush=True)
            return None
        print(f"[rtsp_streaming] started mediamtx PID={proc.pid} on :{port}")
        return proc
    except OSError as exc:
        print(f"[rtsp_streaming] failed to start mediamtx: {exc}")
        return None
