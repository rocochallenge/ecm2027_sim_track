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

"""
Fixture asset generalization utility class

Provides functionality to identify placeholders from USD scenes, acquire random assets via SDK, 
replace placeholders, and adaptively scale assets.
"""

import os
import random
from typing import List, Dict, Any, Optional

# USD library
try:
    from pxr import Usd, UsdGeom, Sdf, Gf
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False
    print("⚠️  Warning: USD Python library (pxr) not available")

SDK_AVAILABLE = False


class FixtureGeneralization:
    """Fixture asset generalization utility class"""

    # Mapping from asset type to registry names
    TYPE_TO_REGISTRIES = {
        "Fridge": ["fridge_bottom_freezer", "fridge_french_door", "fridge_side_by_side"],
        "FridgeBottomFreezer": ["fridge_bottom_freezer"],
        "FridgeFrenchDoor": ["fridge_french_door"],
        "FridgeSideBySide": ["fridge_side_by_side"],
        "FridgeTopFreezer": ["fridge_top_freezer"],
        "Oven": ["oven"],
        "Microwave": ["microwave"],
        "Dishwasher": ["dishwasher"],
        "SingleCabinet": ["cabinet"],
        "Stove": ["stove"],
        "Sink": ["sink"],
        "Counter": ["counter"],
    }

    def __init__(self, stage):
        """
        Args:
            stage: USD Stage 对象
        """
        self.stage = stage
        if not USD_AVAILABLE:
            raise RuntimeError("USD Python library (pxr) is required")

    def find_placeholders(self, root_path: str = "/World", fixture_types: Optional[List[str]] = None) -> List[Dict]:
        """
        Find placeholders in the scene

        Args:
            root_path: Root path to search
            fixture_types: Specify fixture types to find, None means find all types

        Returns:
            List of placeholder information, each placeholder contains:
            - path: prim path
            - name: prim name
            - asset_type: Asset type (e.g., "Dishwasher", "FridgeBottomFreezer")
            - position: Position [x, y, z]
            - rotation: Rotation [x, y, z]
            - scale: Scale [x, y, z]
            - size: Original size [x, y, z]
        """
        placeholders = []
        root_prim = self.stage.GetPrimAtPath(root_path)

        if not root_prim:
            return placeholders

        for prim in self.stage.Traverse():
            if not prim.IsValid():
                continue

            prim_name = prim.GetName()
            prim_type = prim.GetTypeName()

            # Check if it is a placeholder (Xform type with type and size attributes)
            if self._is_placeholder(prim, prim_name):
                ph_info = self._extract_placeholder_info(prim, prim_name)

                # If type list is specified, only return matching placeholders
                if fixture_types is None or ph_info["asset_type"] in fixture_types:
                    placeholders.append(ph_info)

        return placeholders

    def _is_placeholder(self, prim, prim_name: str) -> bool:
        """Check if prim is a placeholder"""
        # Must be Xform type
        if prim.GetTypeName() != "Xform":
            return False

        # Must have type attribute (asset type identifier)
        type_attr = prim.GetAttribute("type")
        if not type_attr or not type_attr.Get():
            return False

        # Must have size attribute (placeholder characteristic)
        size_attr = prim.GetAttribute("size")
        if not size_attr or not size_attr.Get():
            return False

        # Must have a Mesh child with the same name
        has_mesh_child = False
        for child in prim.GetChildren():
            if child.GetName() == prim_name and child.GetTypeName() == "Mesh":
                has_mesh_child = True
                break

        return has_mesh_child

    def _extract_placeholder_info(self, prim, prim_name: str) -> Dict[str, Any]:
        """Extract information from placeholder prim"""
        prim_path = str(prim.GetPath())

        # Get asset type
        type_attr = prim.GetAttribute("type")
        asset_type = type_attr.Get() if type_attr else "Unknown"

        # Get size attribute (string type, format is "x,y,z")
        size_attr = prim.GetAttribute("size")
        size = None
        if size_attr:
            size_str = size_attr.Get()
            if size_str and isinstance(size_str, str):
                try:
                    size_parts = size_str.split(",")
                    if len(size_parts) == 3:
                        size = [float(size_parts[0]), float(size_parts[1]), float(size_parts[2])]
                except (ValueError, IndexError):
                    pass

        # Get transform information (from Mesh child with the same name)
        position = [0.0, 0.0, 0.0]
        rotation = [0.0, 0.0, 0.0]
        scale = [1.0, 1.0, 1.0]

        for child in prim.GetChildren():
            if child.GetName() == prim_name and child.GetTypeName() == "Mesh":
                # Position
                trans_attr = child.GetAttribute("xformOp:translate")
                if trans_attr:
                    trans = trans_attr.Get()
                    if trans is not None:
                        try:
                            position = [float(trans[i]) for i in range(3)]
                        except (TypeError, ValueError, IndexError):
                            pass

                # Rotation
                rot_attr = child.GetAttribute("xformOp:rotateZYX")
                if rot_attr:
                    rot = rot_attr.Get()
                    if rot is not None:
                        try:
                            rotation = [float(rot[i]) for i in range(3)]
                        except (TypeError, ValueError, IndexError):
                            pass

                # Scale
                scale_attr = child.GetAttribute("xformOp:scale")
                if scale_attr:
                    scale_val = scale_attr.Get()
                    if scale_val is not None:
                        try:
                            scale = [float(scale_val[i]) for i in range(3)]
                        except (TypeError, ValueError, IndexError):
                            pass

                break  # Found matching Mesh, exit loop

        return {
            "path": prim_path,
            "name": prim_name,
            "asset_type": asset_type,
            "position": position,
            "rotation": rotation,
            "scale": scale,
            "size": size,
        }

    def acquire_random_asset(self, asset_type: str) -> Optional[str]:
        """
        Acquire random asset of specified type

        Args:
            asset_type: Asset type (e.g., "Dishwasher", "FridgeBottomFreezer")

        Returns:
            Asset USD file path, returns None on failure
        """
        if not SDK_AVAILABLE:
            print(f"⚠️  SDK not available, cannot acquire asset: {asset_type}")
            return None

        try:
            # Get registry names
            registry_names = self.TYPE_TO_REGISTRIES.get(asset_type, [asset_type.lower().replace(" ", "_")])

            if not registry_names:
                print(f"⚠️  No registry mapping for asset type: {asset_type}")
                return None

            # Randomly select a registry
            selected_registry = random.choice(registry_names)

            # Call SDK to acquire asset (corrected parameter format)
            asset_path, asset_name, asset_res = object_loader.acquire_by_registry(
                "fixtures",
                registry_name=[selected_registry],
                source=["lightwheel"]
            )

            if asset_path and os.path.exists(asset_path):
                return str(asset_path)
            else:
                print(f"⚠️  Failed to acquire asset for type: {asset_type}")
                return None

        except Exception as e:
            print(f"⚠️  Error acquiring asset for {asset_type}: {e}")
            return None

    def get_fixed_asset(self, fixture_name: str) -> Optional[str]:
        """
        Get fixed asset (for fixed strategy)

        Args:
            fixture_name: Fixture name

        Returns:
            Asset USD file path, returns None on failure
        """
        # Fixed mapping logic can be implemented here
        # For example: Dishwasher -> specific asset path
        # Currently uses random strategy
        return None

    def acquire_by_category(self, asset_type: str) -> Optional[str]:
        """
        Acquire asset by category (for category_based strategy)

        Args:
            asset_type: Asset type

        Returns:
            Asset USD file path, returns None on failure
        """
        # Currently uses random strategy
        return self.acquire_random_asset(asset_type)

    def get_asset_bbox(self, asset_path: str) -> Optional[List[float]]:
        """
        Get asset bounding box size (using OpenUsd.get_prim_size)

        Args:
            asset_path: Asset USD file path

        Returns:
            Size [width, height, depth], returns None on failure
        """
        if not USD_AVAILABLE or not os.path.exists(asset_path):
            return None

        try:
            # Temporarily open asset file
            asset_stage = Usd.Stage.Open(asset_path)
            if not asset_stage:
                return None

            # Get default prim or root prim
            root_prim = asset_stage.GetDefaultPrim()
            if not root_prim:
                root_prim = asset_stage.GetPseudoRoot()

            # Use OpenUsd.get_prim_size method (project-wide encapsulated ComputeWorldBound)
            from robofinals.utils.usd_utils import OpenUsd
            size = OpenUsd.get_prim_size(root_prim)

            if size is None:
                return None

            # Convert to list format
            size_list = [float(size[0]), float(size[1]), float(size[2])]

            # Return None if size is zero
            if size_list[0] <= 0 or size_list[1] <= 0 or size_list[2] <= 0:
                return None

            return size_list

        except Exception as e:
            print(f"⚠️  Error getting asset bbox: {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_scale(
        self,
        placeholder_size: Optional[List[float]],
        asset_size: List[float],
    ) -> List[float]:
        """
        Calculate scale ratio to fit asset to placeholder size

        Args:
            placeholder_size: Placeholder size [x, y, z]
            asset_size: Asset size [x, y, z]

        Returns:
            Scale ratio [sx, sy, sz]
        """
        if not placeholder_size or not asset_size:
            return [1.0, 1.0, 1.0]

        return [
            placeholder_size[0] / asset_size[0] if asset_size[0] > 0 else 1.0,
            placeholder_size[1] / asset_size[1] if asset_size[1] > 0 else 1.0,
            placeholder_size[2] / asset_size[2] if asset_size[2] > 0 else 1.0,
        ]

    def replace_placeholder(
        self,
        prim_path: str,
        asset_path: str,
        position: List[float],
        rotation: List[float],
        scale: List[float],
        size: Optional[List[float]] = None,
        size_decimal_places: int = 4,
    ) -> bool:
        """
        Replace placeholder with asset

        Args:
            prim_path: Placeholder prim path
            asset_path: Asset USD file path
            position: Position [x, y, z]
            rotation: Rotation [x, y, z]
            scale: Scale [x, y, z]
            size: Original size [x, y, z] (for saving size attribute)
            size_decimal_places: Decimal places for size attribute

        Returns:
            Success status
        """
        try:
            # Get original prim
            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim:
                print(f"❌ Prim not found: {prim_path}")
                return False

            # Save type attribute
            type_attr = prim.GetAttribute("type")
            type_value = type_attr.Get() if type_attr else None

            # Remove original prim
            self.stage.RemovePrim(prim_path)

            # Create new Xform prim
            new_prim = self.stage.DefinePrim(prim_path, "Xform")

            # Set transform
            xformable = UsdGeom.Xformable(new_prim)

            # Translation
            translate_op = xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(position[0], position[1], position[2]))

            # Rotation
            if rotation and rotation != [0.0, 0.0, 0.0]:
                rotate_op = xformable.AddRotateZYXOp()
                rotate_op.Set(Gf.Vec3d(rotation[0], rotation[1], rotation[2]))

            # Scale
            scale_op = xformable.AddScaleOp()
            scale_op.Set(Gf.Vec3d(scale[0], scale[1], scale[2]))

            # Add asset reference
            new_prim.GetReferences().AddReference(asset_path)

            # Restore type attribute
            if type_value:
                type_attr = new_prim.CreateAttribute("type", Sdf.ValueTypeNames.String)
                type_attr.Set(type_value)

            # Save size attribute
            if size:
                size_str = f"{size[0]:.{size_decimal_places}f},{size[1]:.{size_decimal_places}f},{size[2]:.{size_decimal_places}f}"
                size_attr = new_prim.CreateAttribute("size", Sdf.ValueTypeNames.String)
                size_attr.Set(size_str)

            return True

        except Exception as e:
            print(f"❌ Error replacing placeholder: {e}")
            import traceback
            traceback.print_exc()
            return False
