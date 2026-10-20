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

import os
from typing import Dict

from omegaconf import DictConfig, OmegaConf


class ChallengeGarmentLoader:
    """Load garment configuration files from a Challenge_Garment asset tree."""

    def __init__(self, base_path: str = "Assets/objects/Challenge_Garment"):
        self.base_path = os.path.join(os.getcwd(), base_path)
        self.garment_type_map: Dict[str, str] = {
            "Top_Long": "Top_Long",
            "Top_Short": "Top_Short",
            "Pant_Short": "Pant_Short",
            "Pant_Long": "Pant_Long",
        }

    def load_garment_config(
        self,
        garment_name: str,
        version: str = "Release",
    ) -> DictConfig:
        garment_type = self._get_garment_type(garment_name)
        garment_dir = os.path.join(self.base_path, version, garment_type, garment_name)

        if not os.path.isdir(garment_dir):
            raise FileNotFoundError(f"Garment directory not found: {garment_dir}")

        filepath = None
        for filename in os.listdir(garment_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(garment_dir, filename)
                break

        if filepath is None:
            raise ValueError(
                f"No JSON configuration file found in directory: {garment_dir}"
            )

        return OmegaConf.load(filepath)

    def get_garment_type(self, garment_name: str) -> str:
        mapped_type = self._get_garment_type(garment_name)
        if mapped_type == "Top_Long":
            return "top-long-sleeve"
        if mapped_type == "Top_Short":
            return "top-short-sleeve"
        if mapped_type == "Pant_Short":
            return "short-pant"
        if mapped_type == "Pant_Long":
            return "long-pant"
        raise ValueError(
            f"Unknown mapped garment type '{mapped_type}' for garment name '{garment_name}'."
        )

    def _get_garment_type(self, garment_name: str) -> str:
        parts = garment_name.split("_")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid garment name format: {garment_name}. "
                f"Expected format: 'Type_Length_Seen/Unseen_Index'"
            )

        garment_key = f"{parts[0]}_{parts[1]}"
        if garment_key not in self.garment_type_map:
            raise ValueError(
                f"Unknown garment type: {garment_key}. "
                f"Valid types: {list(self.garment_type_map.keys())}"
            )
        return self.garment_type_map[garment_key]
