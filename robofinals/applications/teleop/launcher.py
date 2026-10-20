#!/usr/bin/env python3
import argparse
import ast
import os
import re
import sys
import yaml
import subprocess
import tempfile
import time
import shutil
import traceback
import csv
import random
import secrets
import threading
import zipfile
import shutil
from pathlib import Path

import cv2
from termcolor import colored

current_dir = Path(os.getcwd())

from robofinals import CONFIGS_PATH  # noqa: E402
from robofinals.utils.video_recorder import get_video_duration
from robofinals.utils.teleop_utils import download_file

try:
    from robocasa_upload.joylo_uploader import JoyLoUploader
except ImportError:
    print("Warning: robocasa_upload.joylo_uploader not found")
    JoyLoUploader = None

# Pre-compile regex pattern for better performance
TIMESTAMP_VIDEO_PATTERN = re.compile(r'^\d{8}_\d{4}\.mp4$')


def _csv_optional_vec_literal(cell: str | None):
    """Parse CSV cell as ``ast.literal_eval`` or treat ``none`` / empty as missing."""
    s = (cell or "").strip()
    if s.lower() in ("none", ""):
        return None
    return ast.literal_eval(s)


def load_layout_task_mapping(csv_file_path):
    """
    Load the mapping of robot-task combinations to layout configurations from a CSV file

    Args:
        csv_file_path (str): Path to the CSV file

    Returns:
        dict: A dictionary mapping (robot, task) tuples to lists of layout configurations,
              each configuration containing:
            - robot: Robot type
            - layout: Layout ID
            - init_robot_base_pos: Robot initial position
            - init_robot_base_ori: Robot initial orientation
            - object_init_offset: Object initialization offset
    """
    robot_task_to_layouts = {}

    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                robot = row['robot']
                task = row['task']
                layout = row['layout']
                try:
                    object_offset = ast.literal_eval(row['object_init_offset'])

                    robot_pos = _csv_optional_vec_literal(row['init_robot_base_pos'])
                    robot_ori = _csv_optional_vec_literal(row['init_robot_base_ori'])

                except (ValueError, SyntaxError, TypeError) as e:
                    print(colored(f"Warning: Failed to parse parameters {row}: {e}", "yellow"))
                    continue

                layout_config = {
                    'robot': robot,
                    'layout': layout,
                    'object_init_offset': object_offset
                }

                if robot_pos is not None:
                    layout_config['init_robot_base_pos'] = robot_pos
                if robot_ori is not None:
                    layout_config['init_robot_base_ori'] = robot_ori

                robot_task_key = (robot, task)
                if robot_task_key not in robot_task_to_layouts:
                    robot_task_to_layouts[robot_task_key] = []
                robot_task_to_layouts[robot_task_key].append(layout_config)

    except FileNotFoundError:
        print(colored(f"warning: file {csv_file_path} is not exist", "yellow"))
        return {}
    except Exception as e:
        print(colored(f"error in loading file: {e}", "red"))
        return {}

    return robot_task_to_layouts


def get_image_tag():
    """
    Get the image tag from environment variable IMAGE_TAG.

    Returns:
        str: The image tag from IMAGE_TAG environment variable, or 'unknown' if not set
    """
    return os.environ.get('IMAGE_TAG', 'unknown')


def get_random_layout_for_task(task_name, robot_type=None, csv_file_path=None):
    """
    Get a random layout configuration for a given robot-task combination.

    Args:
        task_name (str): Task name
        robot_type (str): Robot type
        csv_file_path (str): Path to the CSV file, if None then use the default path

    Returns:
        dict: A random layout configuration, including:
            - layout: Layout ID
            - object_init_offset: Object initialization offset
            - init_robot_base_pos: Robot initial position (if available for the robot type)
            - init_robot_base_ori: Robot initial orientation (if available for the robot type)
        If not found, return the default configuration
    """
    if csv_file_path is None:
        csv_file_path = CONFIGS_PATH / "layout_task_mapping" / "layout_task_mapping.csv"

    if robot_type is None:
        print(colored("warning: No robot type specified, layout configuration cannot be selected", "red"))
        return None

    robot_task_to_layouts = load_layout_task_mapping(csv_file_path)
    robot_task_key = (robot_type, task_name)

    if robot_task_key in robot_task_to_layouts:
        layout_configs = robot_task_to_layouts[robot_task_key]
        selected_config = secrets.choice(layout_configs)
        available_layouts = [config['layout'] for config in layout_configs]

        print(colored(f"robot '{robot_type}' + task '{task_name}' Available layouts: {available_layouts}", "cyan"))
        print(colored(f"Randomly select layout: {selected_config['layout']}", "green"))
        print(colored(f"Object initialization offset: {selected_config['object_init_offset']}", "cyan"))

        if 'init_robot_base_pos' in selected_config:
            print(colored(f"robot Initial position: {selected_config['init_robot_base_pos']}", "cyan"))
        else:
            print(colored(f"robot '{robot_type}' ", "yellow"))

        if 'init_robot_base_ori' in selected_config:
            print(colored(f"robot初始方向: {selected_config['init_robot_base_ori']}", "cyan"))
        else:
            print(colored(f"robot '{robot_type}' No initial positional parameters are used", "yellow"))

        return_config = {
            'layout': selected_config['layout'],
            'object_init_offset': selected_config['object_init_offset']
        }

        if 'init_robot_base_pos' in selected_config:
            return_config['init_robot_base_pos'] = selected_config['init_robot_base_pos']
        if 'init_robot_base_ori' in selected_config:
            return_config['init_robot_base_ori'] = selected_config['init_robot_base_ori']

        return return_config
    else:
        print(colored(f"Warning: Robot not found '{robot_type}' + task '{task_name}' Corresponding layout, use the default layout", "yellow"))
        default_config = {
            'layout': 'robocasakitchen',
            'object_init_offset': [0.0, 0.0],
        }

        return default_config


def pick_random_factory_xgrid_usd(root: Path | None = None) -> str | None:
    """
    Under ``root`` (default: cwd), collect ``Xgrid_Factory_*.usd`` files inside
    directories whose names start with ``Factory``, then return one path at random.

    Matches layouts such as ``Factory022_GS_Scene/Xgrid_Factory_022.usd``.
    """
    root = Path(os.getcwd()) if root is None else Path(root)
    candidates: list[Path] = []
    try:
        for child in sorted(root.iterdir()):
            if not child.is_dir() or not child.name.startswith("Factory"):
                continue
            for p in sorted(child.glob("Xgrid_Factory_*.usd")):
                if p.is_file():
                    candidates.append(p.resolve())
    except OSError as e:
        print(colored(f"Warning: could not scan {root} for factory USD layouts: {e}", "yellow"))
        return None
    if not candidates:
        return None
    # Use ``secrets`` so choice is not biased by imported libs that call ``random.seed``.
    chosen = secrets.choice(candidates)
    print(colored(
        f"Factory USD layout pool: {len(candidates)} file(s) under {root.resolve()}",
        "cyan",
    ))
    return str(chosen)


def resolve_teleop_layout_path(args) -> str:
    """
    If ``--layout_path`` is set, use it. Otherwise try a random
    ``Factory*/Xgrid_Factory_*.usd`` under cwd; if none, use yaml ``layout`` or
    legacy default (factory / kitchen string ids).
    """
    lp = getattr(args, "layout_path", None)
    if lp is not None and str(lp).strip():
        return str(lp).strip()
    picked = pick_random_factory_xgrid_usd(Path(os.getcwd()))
    if picked:
        print(colored(f"No --layout_path: random factory USD -> {picked}", "green"))
        return picked
    ly = getattr(args, "layout", None)
    if ly is not None and str(ly).strip():
        print(colored(
            "No --layout_path and no Factory*/Xgrid_Factory_*.usd under cwd; using yaml layout",
            "yellow",
        ))
        return str(ly).strip()
    fallback = "/FactoryTask001002/FactoryTask001.usd"
    print(colored(f"No layout available; fallback -> {fallback}", "yellow"))
    return fallback


class TeleopDaemon:

    def __init__(self, params, on_exit_callback=None, uploader=None, task_name=None,
                 instance_id=None):
        self.params = params
        # Add image tag to params
        if 'image_tag' not in self.params:
            self.params['image_tag'] = get_image_tag()
        self.on_exit_callback = on_exit_callback
        self.process = None
        self.config_path = None
        self.temp_files = []
        self.is_running = False
        self.exit_handled = False
        self.uploader = uploader
        self.task_name = task_name
        self.instance_id = instance_id
        self.upload_success_handled = False

    def _get_dataset_file_path(self):
        """get dataset file path"""
        original_path = self.params.get('dataset_file', './datasets/dataset.hdf5')

        if original_path.startswith('./datasets/'):
            return original_path.replace('./datasets/', './lwlab/datasets/', 1)

        elif original_path.startswith('datasets/'):
            return original_path.replace('datasets/', 'lwlab/datasets/', 1)

        return original_path

    def create_config(self):
        default_config_path = CONFIGS_PATH / 'data_collection' / 'teleop' / 'teleop_base.yml'
        with open(default_config_path, 'r') as f:
            config = yaml.safe_load(f)

        cleaned_params = {}
        for key, value in self.params.items():
            if isinstance(value, Path):
                cleaned_params[key] = str(value)
            else:
                cleaned_params[key] = value

        config.update(cleaned_params)
        temp_fd, temp_yaml_path = tempfile.mkstemp(suffix='.yml', prefix='teleop_')
        temp_config_name = Path(temp_yaml_path).stem

        with os.fdopen(temp_fd, 'w') as f:
            yaml.dump(config, f,
                      default_flow_style=False,
                      indent=2,
                      sort_keys=False,
                      allow_unicode=True)

        config_dir = CONFIGS_PATH / 'data_collection' / 'teleop'
        final_config_path = config_dir / f'{temp_config_name}.yml'

        shutil.copy2(temp_yaml_path, final_config_path)
        print(f"final_config_path: {final_config_path}")
        self.temp_files = [temp_yaml_path, final_config_path]
        self.config_path = final_config_path

        return temp_config_name, config

    def start(self):
        """Start the teleop_main process"""
        # New session starts; reset upload state for the upcoming task
        self.upload_success_handled = False
        temp_config_name, config = self.create_config()
        # teleop_main_path = Path(__file__).parent / 'teleop_main_private.py'
        teleop_main_path = Path(__file__).parent / 'entrypoint.py'
        print(f"teleop_main_path: {teleop_main_path}")
        cmd = [
            sys.executable,
            str(teleop_main_path),
            f'--task_config={temp_config_name}',
            # '--experience',
            # 'isaaclab.python.kit',
        ]
        if self.params.get('viz') is not None:
            cmd.extend(['--viz', str(self.params.get('viz'))])

        if self.params.get('enable_camera') or self.params.get('enable_cameras'):
            cmd.append('--enable_cameras')

        if self.params.get('headless'):
            cmd.append('--headless')

        physics_backend = self.params.get('physics_backend')
        if physics_backend:
            cmd.extend(['--physics_backend', str(physics_backend)])

        print("Start the teleoperation system...")
        print(f"settings: {self.config_path}")
        print(f"task: {config.get('task', 'Unknown')}")
        print(f"robot: {config.get('robot', 'Unknown')}")
        print(f"device: {config.get('teleop_device', 'Unknown')}")
        if config.get('record'):
            print(f"record to: {config.get('dataset_file', 'Unknown')}")
        print(f"command: {' '.join(cmd)}")
        print(f"teleop_launcher working directory: {os.getcwd()}")
        print(f"teleop_main will run in: {os.getcwd()} (same as launcher)")
        print("-" * 50)

        self.process = subprocess.Popen(cmd, cwd=os.getcwd())
        self.is_running = True

        return self.process.pid

    def _handle_exit(self, return_code=None, check_upload=True):
        """Handle the exit of the process"""
        if self.exit_handled:
            return

        self.exit_handled = True

        # If upload via shared memory signal has already been handled, skip prompt/upload
        if (
            getattr(self, "upload_success_handled", False)
            or self.params.get("launcher_managed_upload", False)
        ):
            print(colored("Skipping exit-time upload for launcher-managed collection.", "cyan"))
            self._cleanup()
            return

        dataset_file = None
        video_files = None
        has_dataset = False
        has_videos = False

        if self.params.get('record'):
            dataset_file = self.params.get('dataset_file', './datasets/dataset.hdf5')
            if dataset_file and os.path.exists(dataset_file):
                print(colored(f"Discovery data files: {dataset_file}", "green"))
                file_size = os.path.getsize(dataset_file) / (1024 * 1024)
                print(f"file size: {file_size:.2f} MB")
                has_dataset = True

        if self.params.get('save_video'):
            layout = self.params.get('layout', '')
            robot = self.params.get('robot', '')
            task = self.params.get('task', '')
            video_dir = f'./lwlab/datasets/{layout}_{robot}_{task}'
            if os.path.exists(video_dir):
                video_files_list = list(Path(video_dir).glob('*.mp4'))
                if video_files_list:
                    print(f"Discovery data files: {len(video_files_list)} 个")
                    for vf in video_files_list[:3]:
                        print(colored(f"Discovery data files: {vf}", "green"))
                    if len(video_files_list) > 3:
                        print(colored(f"... has {len(video_files_list) - 3} 个文件", "cyan"))

                    timestamp_videos = [vf for vf in video_files_list
                                        if TIMESTAMP_VIDEO_PATTERN.match(vf.name)]

                    if timestamp_videos:
                        timestamp_videos.sort(key=lambda x: x.name, reverse=True)
                        latest_video = timestamp_videos[0]
                        print(colored(f"Select the video file with the latest timestamp: {latest_video}", "green"))
                    else:
                        latest_video = video_files_list[0]
                        print(colored(f"No timestamp format video found, use: {latest_video}", "yellow"))

                    video_files = str(latest_video)
                    has_videos = True

        if not has_dataset and not has_videos:
            print(colored("No data files or video files were found, so the task was reset directly....", "yellow"))
            self._reset_task()
        else:
            if check_upload:
                print(colored("\nFound the upload configuration and started the upload process...", "green"))
                upload_success = self._upload_data(dataset_file, video_files)
                if not upload_success:
                    print(colored("...Upload failed, reset task", "yellow"))
                    self._reset_task()
            else:
                print(colored("...Upload skipped, reset task", "yellow"))
                self._reset_task()

        self._cleanup()

    def _reset_task(self):
        """Reset task using unified reset function"""
        reset_task(self.uploader)

    def _get_check_result(self, metric_path):
        """Parse metrics.json and extract checker results

        Args:
            metric_path (str): Path to the metrics.json file

        Returns:
            dict: Dictionary with checker names as keys and success (bool) as values
            Example: {'obj_drop_checker': True, 'distance_checker': False}
        """
        import json
        import os

        checkers = {}

        if os.path.exists(metric_path):
            try:
                with open(metric_path, 'r') as mf:
                    metrics = json.load(mf)
                for checker_name, checker_data in metrics.items():
                    if isinstance(checker_data, dict) and 'success' in checker_data:
                        checkers[checker_name] = bool(checker_data['success'])
            except Exception as e:
                print(colored(f"Failed to parse metrics.json: {e}", "yellow"))

        return checkers

    def _get_layout_from_hdf5(self, file_path=None):
        """get layout from hdf5 file"""
        import h5py
        import json
        import os

        if file_path is None:
            file_path = self.params.get('dataset_file', './datasets/dataset_success.hdf5')
        print(f"open file: {file_path}")
        print(f"file exists: {os.path.exists(file_path)}")

        layout_from_hdf5 = None
        hdf5_file_path = file_path
        if os.path.exists(hdf5_file_path):
            # Create temporary file path in the same directory
            file_dir = os.path.dirname(hdf5_file_path)
            file_name = os.path.basename(hdf5_file_path)
            # Generate temp file name: replace .hdf5 with _tmp.hdf5
            if file_name.endswith('.hdf5'):
                temp_file_name = file_name[:-5] + '_tmp.hdf5'
            else:
                temp_file_name = file_name + '_tmp.hdf5'
            temp_file_path = os.path.join(file_dir, temp_file_name) if file_dir else temp_file_name

            try:
                # Copy original file to temporary file
                print(f"copying file to temporary file: {temp_file_path}")
                shutil.copy2(hdf5_file_path, temp_file_path)

                # Read from temporary file instead of original file
                with h5py.File(temp_file_path, 'r') as f:
                    if 'data' in f and 'env_args' in f['data'].attrs:
                        env_args_str = f['data'].attrs['env_args']
                        env_args = json.loads(env_args_str)

                        # check if there is layout_id and style_id
                        layout_id = env_args.get('layout_id')
                        style_id = env_args.get('style_id')

                        if layout_id is not None and style_id is not None:
                            # check if the original layout is robocasakitchen
                            original_layout = self.params.get('layout', '')
                            if original_layout == 'robocasakitchen':
                                layout_from_hdf5 = f"robocasakitchen-{layout_id}-{style_id}"
                                print(colored(f"get layout from hdf5 file: {layout_from_hdf5}", "green"))
                                self.params['layout'] = layout_from_hdf5
                            else:
                                layout_from_hdf5 = original_layout
                                print(colored(f"use original layout: {layout_from_hdf5}", "cyan"))
            except Exception as e:
                print(colored(f"read hdf5 file failed: {e}", "yellow"))
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        print(f"removed temporary file: {temp_file_path}")
                    except Exception as e:
                        print(colored(f"failed to remove temporary file {temp_file_path}: {e}", "yellow"))

    def _upload_data(self, recording_path, video_files):
        """upload data

        Returns:
            bool: True if upload was successful, False otherwise
        """
        print(colored("Uploader configuration detected", "green"))
        user_input = input(colored("Do you want to upload this record? (y/n): ", "cyan"))
        while True:
            if user_input.lower() == 'n':
                print(colored("skip upload", "yellow"))

                return False
            elif user_input.lower() == 'y':
                try:
                    print(colored("start upload...", "green"))

                    running_args = {
                        "task_name": self.task_name,
                        "instance_id": self.instance_id
                    }

                    video_duration = 0.0
                    if video_files and os.path.exists(video_files):
                        video_duration = get_video_duration(video_files)
                        print(colored(f"video len : {video_duration} s", "cyan"))

                    self._get_layout_from_hdf5(str(recording_path))
                    metadata = {
                        "is_success": True,
                        "recording_path": str(recording_path),
                        "params": self.params,
                        "checkpoints": [],
                        "video_duration": video_duration
                    }
                    lerobot_archive = self._archive_lerobot_dataset(recording_path)
                    upload_files = [Path(recording_path),
                                    current_dir / 'lwlab_logs',]
                    if lerobot_archive is not None:
                        upload_files.append(lerobot_archive)
                    result = self.uploader.upload(
                        self.task_name,
                        upload_files,
                        Path(video_files) if video_files is not None else None,
                        2,  # 2: lwlab, 1: joylo
                        running_args,
                        metadata,
                    )

                    print(colored("upload success!", "green"))
                    print(f"upload result: {result}")
                    return True

                except Exception as e:
                    print(colored(f"upload failed: {e}", "red"))
                    print(traceback.format_exc())
                    return False

            else:
                print(colored("Invalid input, please enter y or n:", "red"))
                user_input = input()

    def _lerobot_dataset_ready(self, lerobot_root: Path) -> bool:
        """True when LeRobot v3 parquet writers have been finalized on disk."""
        episodes_dir = lerobot_root / "meta" / "episodes"
        episode_files = sorted(episodes_dir.glob("**/*.parquet")) if episodes_dir.exists() else []
        if not episode_files:
            return False

        data_files = sorted((lerobot_root / "data").glob("**/*.parquet")) if (lerobot_root / "data").exists() else []
        for parquet_path in data_files:
            try:
                if parquet_path.stat().st_size < 8:
                    return False
                if parquet_path.read_bytes()[-4:] != b"PAR1":
                    return False
            except OSError:
                return False
        return True

    def _archive_lerobot_dataset(self, hdf5_path, lerobot_root_override: str | None = None):
        """Zip the LeRobot dataset directory next to the HDF5 file before upload."""
        if not self.params.get('record_lerobot'):
            return None

        hdf5_path = Path(hdf5_path)
        lerobot_root = lerobot_root_override or self.params.get('lerobot_dataset_root')
        if lerobot_root is None or not str(lerobot_root).strip():
            lerobot_root = hdf5_path.parent / hdf5_path.stem
        lerobot_root = Path(lerobot_root)
        if not lerobot_root.exists() or not lerobot_root.is_dir():
            print(colored(f"LeRobot dataset folder not found, skip archive: {lerobot_root}", "yellow"))
            return None

        for attempt in range(30):
            if self._lerobot_dataset_ready(lerobot_root):
                break
            time.sleep(1.0)
        else:
            print(colored(
                f"LeRobot dataset not finalized (missing episode meta or parquet footer): {lerobot_root}",
                "yellow",
            ))
            return None

        episode_files = sorted((lerobot_root / "meta" / "episodes").glob("**/*.parquet"))
        if not episode_files:
            print(colored(
                f"LeRobot episode metadata not found, skip archive: {lerobot_root / 'meta' / 'episodes'}",
                "yellow",
            ))
            return None

        archive_path = hdf5_path.parent / f"{lerobot_root.name}.zip"
        if archive_path.exists():
            archive_path.unlink()
        print(colored(f"Archiving LeRobot dataset: {lerobot_root} -> {archive_path}", "cyan"))

        def should_skip(path: Path) -> bool:
            rel_parts = path.relative_to(lerobot_root).parts
            return any(part.startswith("tmp") for part in rel_parts)

        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(lerobot_root.rglob("*")):
                if not file_path.is_file() or should_skip(file_path):
                    continue
                zf.write(
                    file_path,
                    arcname=str(Path(lerobot_root.name) / file_path.relative_to(lerobot_root)),
                )

        return archive_path

    def _upload_success_data(self, success_hdf5_path, lerobot_root_override: str | None = None):
        """Upload success data automatically without user input

        Args:
            success_hdf5_path (str): Path to the success hdf5 file
            lerobot_root_override (str | None): LeRobot dataset root from teleop signal

        Returns:
            bool: True if upload was successful, False otherwise
        """
        try:
            print(colored("Auto-uploading success data...", "green"))

            running_args = {
                "task_name": self.task_name,
                "instance_id": self.instance_id
            }

            # Get directory path of success_hdf5_path and append metrics.json
            from pathlib import Path
            metrics_path = Path(success_hdf5_path).parent / "metrics.json"
            checkers = self._get_check_result(str(metrics_path))
            print(f"checkers: {checkers}")

            # Get layout from hdf5 file
            self._get_layout_from_hdf5(str(success_hdf5_path))
            metadata = {
                "is_success": True,
                "recording_path": str(success_hdf5_path),
                "params": self.params,
                "checkpoints": [],
                "checkers": checkers,
                "video_duration": 0.0
            }

            lerobot_archive = self._archive_lerobot_dataset(
                success_hdf5_path,
                lerobot_root_override=lerobot_root_override,
            )
            upload_files = [Path(success_hdf5_path),
                            metrics_path,
                            current_dir / 'lwlab_logs']
            if lerobot_archive is not None:
                upload_files.append(lerobot_archive)

            result = self.uploader.upload(
                self.task_name,
                upload_files,
                None,  # No video files for success data
                2,  # 2: lwlab, 1: joylo
                running_args,
                metadata,
            )

            print(colored("Success data upload completed!", "green"))
            print(f"Upload result: {result}")

            # Mark that upload has been handled to avoid exit-time prompt/upload
            self.upload_success_handled = True

            return True

        except Exception as e:
            print(colored(f"Success data upload failed: {e}", "red"))
            print(traceback.format_exc())
            return False

    def _pull_and_signal_next_task(self):
        """Pull the next launcher task after prior-cycle files have been cleaned up."""
        try:
            print(colored("Re-pulling task after successful upload...", "cyan"))
            new_task_info = self.uploader.pull_task(batch_name=self.params.get("batch_name"))
            if new_task_info is not None:
                print(colored(f"New task pulled: {new_task_info.get('env_name', 'Unknown')}", "green"))
                self.task_name = new_task_info.get("env_name")
                self.params["task"] = self.task_name
                scene_backend = str(self.params.get("scene_backend", "robocasa")).lower()
                task_backend = self.params.get("task_backend", "robocasa")
                if scene_backend == "local":
                    task_config = {
                        "layout": self.params.get("layout") or self.params.get("scene_name"),
                        "object_init_offset": self.params.get("object_init_offset", [0.0, 0.0]),
                        "scene_backend": scene_backend,
                        "task_backend": task_backend,
                    }
                    for key in (
                        "init_robot_base_pos",
                        "init_robot_base_ori",
                    ):
                        if key in self.params:
                            task_config[key] = self.params[key]
                else:
                    task_config = get_random_layout_for_task(
                        self.task_name,
                        self.params.get("robot"),
                    ) or {}
                    task_config.update(
                        {
                            "scene_backend": scene_backend,
                            "task_backend": task_backend,
                        }
                    )
                next_seed = int(self.params.get("seed", 0)) + 1
                task_config["seed"] = next_seed
                print(colored(f"Next collection seed: {next_seed}", "cyan"))
                for key in (
                    "layout",
                    "object_init_offset",
                    "init_robot_base_pos",
                    "init_robot_base_ori",
                    "scene_backend",
                    "task_backend",
                    "seed",
                ):
                    if key in task_config:
                        self.params[key] = task_config[key]
                self._send_task_signal(has_task=True, task_config=task_config)
            else:
                print(colored("No new task available", "yellow"))
                self._send_task_signal(has_task=False)
        except Exception as e:
            print(colored(f"Error re-pulling task: {e}", "yellow"))
            self._send_task_signal(has_task=False)

    def _send_task_signal(self, has_task, task_name=None, task_config=None):
        """send task status signal to teleop_main"""
        try:
            import json
            import time
            from multiprocessing import shared_memory

            if has_task:
                # A new/next task is available; clear upload-completed state for the new run
                self.upload_success_handled = False
                signal_data = {
                    "action": "task_available",
                    "timestamp": time.time(),
                    "message": "New task available, continuing teleoperation",
                    "has_task": True,
                    "task_name": task_name or self.task_name
                }
                if task_config:
                    signal_data["task_config"] = task_config
                color = "green"
            else:
                signal_data = {
                    "action": "no_task_available",
                    "timestamp": time.time(),
                    "message": "No new task available, stopping teleoperation",
                    "has_task": False
                }
                color = "yellow"

            signal_name = self.params.get("teleop_task_signal_name", "teleop_task_signal")
            try:
                shm = shared_memory.SharedMemory(name=signal_name, create=False)
            except FileNotFoundError:
                shm = shared_memory.SharedMemory(name=signal_name, create=True, size=4096)

            signal_json = json.dumps(signal_data)
            signal_bytes = signal_json.encode('utf-8')

            shm.buf[:len(signal_bytes)] = signal_bytes
            shm.buf[len(signal_bytes)] = 0  # Null terminator
            shm.close()

            print(colored(f"Task signal sent: {signal_data['message']}", color))
            return True

        except Exception as e:
            print(f"Error sending task signal: {e}")
            traceback.print_exc()
            return False

    def _cleanup(self):
        """Clean up temporary files"""
        print(colored("\nCleaning up temporary files...", "green"))
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    print(f"delete: {temp_file}")
            except Exception as e:
                print(f"can not delete {temp_file}: {e}")

        # Try to find and kill processes on port 8012
        try:
            result = subprocess.run(['lsof', '-i:8012'], capture_output=True, text=True, check=False, timeout=5)
            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    # Skip header and parse PIDs
                    for line in lines[1:]:
                        parts = line.split()
                        if len(parts) > 1:
                            pid = parts[1]
                            try:
                                print(f"killing PID: {pid}")
                                subprocess.run(['kill', '-9', pid], check=False)
                            except Exception as e:
                                print(f"cannot kill PID:  {pid}: {e}")
        except FileNotFoundError:
            print("warning: 'lsof' command not found. cannot kill process on port 8012.")
        except subprocess.TimeoutExpired:
            print("warning: 'lsof' command timeout. cannot kill process on port 8012.")
        except Exception as e:
            print(f"error: cannot kill process on port 8012: {e}")

    def stop(self, check_upload=True):
        """Stop the teleop process"""
        if self.process and self.is_running:
            print(colored("Stopping teleoperation process...", "red"))
            self.process.terminate()
            try:
                return_code = self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(colored("Forcefully terminate a process...", "red"))
                self.process.kill()
                return_code = -1

            self.is_running = False
            self._handle_exit(return_code, check_upload)

    def is_alive(self):
        """Check if the process is still running"""
        return self.is_running and self.process and self.process.poll() is None


def reset_task(uploader, context=""):
    """
    reset task

    Args:
        uploader: uploader instance
        context (str): context info for log
    """
    try:
        if context:
            print(colored(f"{context}, Resetting task...", "yellow"))
        else:
            print(colored("Resetting task...", "yellow"))
        uploader.update_task(mode="reset")
        print(colored("task reset success", "green"))
    except Exception as e:
        print(colored(f"task reset failed: {e}", "red"))


def str_to_bool(value):
    """Convert string to boolean value."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes', 'on'):
        return True
    elif value.lower() in ('false', '0', 'no', 'off'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got: {value}')


def main():
    """Main function with comprehensive error handling"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default=None)
    parser.add_argument('--batch_name', type=str, default=None)
    parser.add_argument('--output_hdf5_path', type=str, default=None)
    parser.add_argument('--check_upload', type=str_to_bool, default=True)
    parser.add_argument('--enable_camera', '--enable_cameras', dest='enable_camera', type=str_to_bool, default=True)
    parser.add_argument('--scene_group', type=int, default=None)
    parser.add_argument('--robot', type=str, default='X7S-Abs')
    parser.add_argument('--continue_teleop', type=str_to_bool, default=True)
    parser.add_argument('--teleop_device', type=str, default='vr-controller')
    parser.add_argument('--headless', type=str_to_bool, default=False)
    parser.add_argument('--viz', type=str, default=None)
    parser.add_argument('--task_info_file_path', type=str, default=None)
    parser.add_argument('--is_cloud_mode', type=bool, default=False)
    parser.add_argument(
        '--layout_path',
        type=str,
        default=None,
        help='Optional USD scene path; if set, overrides `layout` from task YAML (for local-USD workflows).',
    )
    parser.add_argument(
        '--physics_backend',
        type=str,
        default=None,
        choices=['newton', 'physx'],
        help='Simulation backend forwarded to teleop_main subprocess.',
    )

    parser.add_argument('--task_config', type=str, default='local_teleop')
    parser.add_argument(
        '--num_envs',
        type=int,
        default=None,
        help='Override parallel environment count; omit to use value from task yaml.',
    )
    args = parser.parse_args()
    from robofinals.utils.config_loader import config_loader, merge_task_yaml_with_cli

    yaml_args = config_loader.load(args.task_config)
    for key in ("robot", "layout", "task"):
        yaml_args.__dict__.pop(key, None)
    merge_task_yaml_with_cli(args, yaml_args)

    # Do not overwrite YAML `layout` with a default factory path: that forced LocalScene-style
    # USD loading even when scene_backend/task_backend were robocasa in the YAML.
    if args.layout_path is not None:
        args.layout = args.layout_path

    uploader = None
    task_name = None
    daemon = None
    preprocess_task_mode = False

    try:
        try:
            import json
            if args.task_info_file_path is not None:
                task_info = json.load(open(args.task_info_file_path, 'r', encoding='utf-8'))
                args.task_name = task_info["level1"]
                task_name = args.task_name
                args.robot = task_info["robot_type"]
            else:
                uploader = JoyLoUploader(project_name=args.project_name)
                task_info = uploader.pull_task(batch_name=args.batch_name)
                if task_info is None:
                    raise ValueError(f"Unable to obtain task information: batch_name='{args.batch_name}' May not exist or may have been completed")
                args.task_name = task_info["env_name"]
                if task_info.get("hdf5_url", None) is not None:
                    preprocess_task_mode = True
                    tmp_dir = str(Path(__file__).parent.parent.parent.parent / 'tmp')
                    download_file(task_info["hdf5_url"], tmp_dir, "input_dataset.hdf5")
                    args.input_dataset_file = str(Path(tmp_dir) / "input_dataset.hdf5")
                task_name = args.task_name
            args.instance_id = 1

        except Exception as e:
            print(colored(f"Error: Failed to get task - {str(e)}", "red"))
            raise

        output_dir = Path('./datasets')
        output_dir.mkdir(parents=True, exist_ok=True)
        args.recording_path = str(output_dir / f"{args.task_name}.hdf5")

        layout_config = get_random_layout_for_task(args.task_name, args.robot)
        if layout_config is None:
            raise ValueError(f"Unable to get layout configuration：robot='{args.robot}', task='{args.task_name}'")
        output_hdf5_path = args.output_hdf5_path if args.output_hdf5_path is not None else str(Path(__file__).parent.parent.parent.parent / 'third_party' / 'lwlab' / 'datasets' / 'dataset.hdf5')
        lerobot_dataset_root = str(Path(output_hdf5_path).with_suffix(''))
        if os.path.exists(lerobot_dataset_root):
            print(f"start Dataset directory found: {lerobot_dataset_root}")
            shutil.rmtree(lerobot_dataset_root)
        scene_backend = getattr(args, 'scene_backend', 'robocasa')
        task_backend = getattr(args, 'task_backend', 'robocasa')
        ipc_token = f"{os.getpid()}_{secrets.token_hex(6)}"
        upload_signal_name = f"teleop_upload_signal_{ipc_token}"
        task_signal_name = f"teleop_task_signal_{ipc_token}"
        # Robocasa: use layout from robot–task mapping (floorplan_loader download), not a stale USD path.
        if str(scene_backend).lower() == 'robocasa':
            scene_layout = layout_config['layout']
        else:
            scene_layout = args.layout
        test_params = {
            'scene_backend': scene_backend,
            'task_backend': task_backend,
            'scene_name': scene_layout,
            'teleop_device': args.teleop_device,
            'viz': args.viz,
            'robot': args.robot,
            'task': args.task_name,
            'record': True,
            'dataset_file': output_hdf5_path,
            'record_lerobot': True,
            'lerobot_dataset_root': lerobot_dataset_root,
            'save_video': False,
            'video_save_dir': str(Path(__file__).parent.parent.parent.parent / 'third_party' / 'lwlab' / 'datasets'),
            'layout': scene_layout,
            'num_demos': 1,
            'scene_group': args.scene_group,
            'enable_cameras': args.enable_camera,
            'seed': int(getattr(args, 'seed', 0) or 0),
            'object_init_offset': layout_config['object_init_offset'],
            'first_person_view': True,
            'enable_multiple_viewports': True,
            'enable_rendering_optimization': True,
            'continue_teleop_after_success': args.continue_teleop,
            'launcher_managed_upload': True,
            'headless': args.headless,
            'teleop_upload_signal_name': upload_signal_name,
            'teleop_task_signal_name': task_signal_name,
            'is_cloud_mode': args.is_cloud_mode,
            "device": "cpu",
        }
        if args.physics_backend is not None:
            test_params['physics_backend'] = args.physics_backend
        if args.project_name is not None and args.batch_name is not None:
            test_params['project_name'] = args.project_name
            test_params['batch_name'] = args.batch_name
        if preprocess_task_mode:
            test_params['preprocess_task_mode'] = preprocess_task_mode
        if layout_config and 'init_robot_base_pos' in layout_config:
            test_params['init_robot_base_pos'] = layout_config['init_robot_base_pos']
        if layout_config and 'init_robot_base_ori' in layout_config:
            test_params['init_robot_base_ori'] = layout_config['init_robot_base_ori']
        if hasattr(args, "input_dataset_file"):
            test_params['input_dataset_file'] = args.input_dataset_file
        print(colored("Test start teleoperation daemon...", "green"))
        print(f"params: {test_params}")

        test_params['batch_name'] = args.batch_name
        print(f"Initializing shared memories before starting teleop_main...")
        upload_signal_shm = None
        task_signal_shm = None
        try:
            from multiprocessing import shared_memory
            import json

            try:
                upload_signal_shm = shared_memory.SharedMemory(name=upload_signal_name, create=False)
            except FileNotFoundError:
                upload_signal_shm = shared_memory.SharedMemory(name=upload_signal_name, create=True, size=4096)

            try:
                task_signal_shm = shared_memory.SharedMemory(name=task_signal_name, create=False)
            except FileNotFoundError:
                task_signal_shm = shared_memory.SharedMemory(name=task_signal_name, create=True, size=4096)
        except Exception as e:
            print(f"Error initializing shared memories: {e}")
            traceback.print_exc()
            upload_signal_shm = None
            task_signal_shm = None

        daemon = TeleopDaemon(test_params, None, uploader, args.task_name, args.instance_id)

        # Send initial task signal before starting teleop_main
        print("Sending initial task signal...")
        daemon._send_task_signal(has_task=True, task_name=args.task_name)

        daemon.start()

        print(f"start, PID: {daemon.process.pid}")

        try:
            while daemon.process and daemon.process.poll() is None:
                time.sleep(3)

                if upload_signal_shm is not None:
                    try:
                        data_bytes = bytes(upload_signal_shm.buf).split(b'\x00')[0]
                        if data_bytes:
                            signal_data = json.loads(data_bytes.decode('utf-8'))

                            if signal_data.get("action") == "upload_success":
                                success_hdf5_path = signal_data.get("hdf5_path")
                                has_success_file = signal_data.get("has_success_file", False)
                                signal_lerobot_root = signal_data.get("lerobot_dataset_root") or lerobot_dataset_root

                                if success_hdf5_path and os.path.exists(success_hdf5_path) and has_success_file:
                                    print(colored(f"Upload signal detected via shared memory for: {success_hdf5_path}", "green"))
                                    print(colored(f"Boolean signal: has_success_file = {has_success_file}", "cyan"))

                                    upload_success = daemon._upload_success_data(
                                        success_hdf5_path,
                                        lerobot_root_override=signal_lerobot_root,
                                    )
                                    if upload_success:
                                        print(colored("Upload completed successfully", "green"))
                                        # restore uploaded filename to original filename
                                        try:
                                            original_hdf5_path = success_hdf5_path.replace("_success.hdf5", ".hdf5")
                                            shutil.move(success_hdf5_path, original_hdf5_path)
                                            print(f"Restored dataset file: {success_hdf5_path} -> {original_hdf5_path}")
                                            if signal_lerobot_root and os.path.exists(signal_lerobot_root):
                                                print(f"Removing finalized LeRobot dataset for next cycle: {signal_lerobot_root}")
                                                shutil.rmtree(signal_lerobot_root)
                                        except Exception as e:
                                            print(f"Error restoring dataset file: {e}")
                                        daemon._pull_and_signal_next_task()
                                    else:
                                        print(colored("Upload failed", "red"))

                                    # clear upload signal from shared memory
                                    upload_signal_shm.buf[:] = b'\x00' * len(upload_signal_shm.buf)
                                    print("Cleared upload signal from shared memory")

                            elif signal_data.get("action") == "no_task_available":
                                print(colored("No task signal detected in main loop - preserving signal", "yellow"))
                                pass

                    except Exception as e:
                        print(f"Error processing shared memory signal: {e}")

            if upload_signal_shm is not None:
                try:
                    upload_signal_shm.close()
                    upload_signal_shm.unlink()
                    print("Upload shared memory cleaned up")
                except Exception as e:
                    print(f"Error cleaning up upload shared memory: {e}")

            if task_signal_shm is not None:
                try:
                    task_signal_shm.close()
                    task_signal_shm.unlink()
                    print("Task shared memory cleaned up")
                except Exception as e:
                    print(f"Error cleaning up task shared memory: {e}")

            return_code = daemon.process.returncode if daemon.process else -1
            print("\n" + "=" * 50)
            print(colored(f"teleop stop, return code: {return_code}", "red"))
            daemon._handle_exit(return_code, args.check_upload)

        except KeyboardInterrupt:
            print(colored("\nreceived stop signal...", "red"))
            daemon.stop(args.check_upload)
            print(colored("process stop", "green"))

    except Exception as e:
        print(colored(f"error in process: {e}", "red"))
        print(traceback.format_exc())

        if task_name:
            reset_task(uploader, "error before start")

        if daemon and daemon.is_alive():
            try:
                daemon.stop(args.check_upload)
            except Exception as stop_e:
                print(colored(f"error in stop process: {stop_e}", "red"))

        try:
            if 'upload_signal_shm' in locals() and upload_signal_shm is not None:
                upload_signal_shm.close()
                upload_signal_shm.unlink()
                print("Upload shared memory cleaned up on error")
        except Exception as cleanup_e:
            print(f"Error cleaning up upload shared memory: {cleanup_e}")

        try:
            if 'task_signal_shm' in locals() and task_signal_shm is not None:
                task_signal_shm.close()
                task_signal_shm.unlink()
                print("Task shared memory cleaned up on error")
        except Exception as cleanup_e:
            print(f"Error cleaning up task shared memory: {cleanup_e}")

        sys.exit(1)


if __name__ == "__main__":
    main()
