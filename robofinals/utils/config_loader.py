# Copyright 2025 Lightwheel Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import importlib.metadata
from pathlib import Path

import yaml

from robofinals import CONFIGS_PATH


class ConfigLoader:
    def __init__(self):
        self.configs_root = CONFIGS_PATH
        self.yml_meta = {}
        self._collect_yml_files()

    def _collect_yml_files(self):
        for yml_path in self.configs_root.rglob("*.yml"):
            name = yml_path.stem
            self.yml_meta[name] = yml_path.resolve()
        for yml_path in self.configs_root.rglob("*.yaml"):
            name = yml_path.stem
            self.yml_meta[name] = yml_path.resolve()

        entry_points = importlib.metadata.entry_points()
        robofinals_modules = entry_points.select(group="robofinals_modules")
        for entry_point in robofinals_modules:
            if entry_point.name == "config_search_path":
                additional_config_path = entry_point.load().__path__[0]
                for yml_path in Path(additional_config_path).rglob("*.yml"):
                    name = yml_path.stem
                    self.yml_meta[name] = yml_path.resolve()
                for yml_path in Path(additional_config_path).rglob("*.yaml"):
                    name = yml_path.stem
                    self.yml_meta[name] = yml_path.resolve()
                break

    def load(self, name, loaded_files=None):
        if name not in self.yml_meta:
            raise FileNotFoundError(f"Config file not found: {name}")
        yaml_path = self.yml_meta[name]
        if loaded_files is None:
            loaded_files = set()
        abs_path = yaml_path.resolve()
        if abs_path in loaded_files:
            raise RuntimeError(f"Circular reference detected: {abs_path}")
        loaded_files.add(abs_path)

        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        base_cfg = {}
        if "_base_" in cfg and cfg["_base_"]:
            base_files = cfg["_base_"]
            if not isinstance(base_files, list):
                base_files = [base_files]
            for base_file in base_files:
                if isinstance(base_file, str):
                    base_name = Path(base_file).stem
                    if base_name in self.yml_meta:
                        base_file_path = self.yml_meta[base_name]
                    else:
                        base_file_path = (yaml_path.parent / base_file).resolve()
                        if not base_file_path.exists():
                            raise FileNotFoundError(f"Base config file not found: {base_file}")
                        self.yml_meta[base_name] = base_file_path
                else:
                    raise ValueError(f"_base_ only supports string: {base_file}")
                base_cfg_part = self.load(base_name, loaded_files)
                if isinstance(base_cfg_part, argparse.Namespace):
                    base_cfg_part = vars(base_cfg_part)
                base_cfg = self.merge_dicts(base_cfg, base_cfg_part)
            cfg.pop("_base_")

        merged = self.merge_dicts(base_cfg, cfg)
        return argparse.Namespace(**merged)

    @staticmethod
    def merge_dicts(base, override):
        result = dict(base)
        for k, v in override.items():
            if (
                k in result
                and isinstance(result[k], dict)
                and isinstance(v, dict)
            ):
                result[k] = ConfigLoader.merge_dicts(result[k], v)
            else:
                result[k] = v
        return result


def merge_task_yaml_with_cli(
    args_cli: argparse.Namespace,
    yaml_args: argparse.Namespace,
    *,
    override_keys: tuple[str, ...] = ("num_envs", "physics_backend"),
) -> None:
    """将 *yaml_args* 合并进 *args_cli*。对 *override_keys* 中在命令行显式给出的非 None 值保留 CLI 结果。"""
    saved = {k: getattr(args_cli, k) for k in override_keys if hasattr(args_cli, k)}
    args_cli.__dict__.update(yaml_args.__dict__)
    for k, v in saved.items():
        if v is not None:
            setattr(args_cli, k, v)


config_loader = ConfigLoader()
