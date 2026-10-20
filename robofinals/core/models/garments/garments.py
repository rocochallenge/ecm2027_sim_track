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

from __future__ import annotations

from typing import Any

import numpy as np
import torch

ENABLE_CHECKPOINT_DEBUG_VISUALS = True
CHECK_SUCCESS_DEBUG_ROOT_PATH = "/World/Debug/FoldGarmentCheckpoints"
CHECK_SUCCESS_DEBUG_COLORS = [
    (1.0, 0.2, 0.2),
    (1.0, 0.6, 0.2),
    (1.0, 0.9, 0.2),
    (0.2, 0.9, 1.0),
    (0.2, 0.5, 1.0),
    (0.7, 0.2, 1.0),
]


def garment_point_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_a - point_b))


def check_top_sleeve_fold(points: np.ndarray, thresholds: list[float]) -> bool:
    dist_0_4 = garment_point_distance(points[0], points[4])
    dist_2_3 = garment_point_distance(points[2], points[3])
    dist_1_5 = garment_point_distance(points[1], points[5])
    dist_0_1 = garment_point_distance(points[0], points[1])
    dist_4_5 = garment_point_distance(points[4], points[5])
    cond1 = dist_0_4 <= thresholds[0]
    cond2 = dist_2_3 <= thresholds[1]
    cond3 = dist_1_5 <= thresholds[2]
    cond4 = dist_0_1 >= thresholds[3]
    cond5 = dist_4_5 >= thresholds[4]
    return cond1 and cond2 and cond3 and cond4 and cond5


class GarmentModel:
    def __init__(self, garment: Any):
        self.garment = garment

    def get_scaled_success_thresholds(self) -> list[float]:
        scale = float(self.garment.init_scale[0])
        return [float(value) * scale for value in self.garment.success_distance]

    def get_checkpoint_positions_cm(self) -> np.ndarray | None:
        positions = self.garment._cloth_prim_view.get_world_positions()
        if not torch.is_tensor(positions):
            positions = torch.as_tensor(positions, dtype=torch.float32)
        positions = positions.detach().float().reshape(-1, 3).cpu().numpy()

        check_points = [int(index) for index in self.garment.check_points]
        if positions.shape[0] <= max(check_points):
            return None

        sampled = positions[np.asarray(check_points, dtype=np.int64)] * 100.0
        if not np.isfinite(sampled).all():
            return None
        return sampled

    def update_checkpoint_debug_visuals(
        self,
        env: Any,
        checkpoint_positions_cm: np.ndarray,
        debug_root_path: str = CHECK_SUCCESS_DEBUG_ROOT_PATH,
    ) -> None:
        if not ENABLE_CHECKPOINT_DEBUG_VISUALS:
            return

        try:
            from pxr import Gf, UsdGeom
        except ModuleNotFoundError:
            return

        stage = env.scene.stage
        debug_root = stage.GetPrimAtPath("/World/Debug")
        if not debug_root.IsValid():
            UsdGeom.Xform.Define(stage, "/World/Debug")

        UsdGeom.Xform.Define(stage, debug_root_path)
        checkpoint_positions_m = checkpoint_positions_cm / 100.0
        for idx, point in enumerate(checkpoint_positions_m):
            sphere_path = f"{debug_root_path}/p{idx}"
            sphere = UsdGeom.Sphere.Define(stage, sphere_path)
            sphere.CreateRadiusAttr(0.015)
            xform = UsdGeom.Xformable(sphere.GetPrim())
            translate_ops = [
                op for op in xform.GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
            ]
            translate_op = translate_ops[0] if translate_ops else xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
            color_values = CHECK_SUCCESS_DEBUG_COLORS[idx] if idx < len(CHECK_SUCCESS_DEBUG_COLORS) else (1.0, 1.0, 1.0)
            sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*color_values)])
