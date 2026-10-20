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

"""Backend-neutral material conversions used by deformable USD adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass


# Newton 1.4 uses the same fallback when a surface material contains volumetric
# values but does not author a shell thickness.
DEFAULT_SURFACE_THICKNESS = 0.002


@dataclass(frozen=True)
class SurfaceMaterialInputs:
    """OmniPhysics surface material values before Newton conversion.

    Density and stiffness inputs are volumetric quantities. ``None`` and zero
    stiffness overrides both mean that the optional surface override is not
    active, matching ``OmniPhysicsSurfaceDeformableMaterialAPI``.
    """

    thickness: float | None = None
    density: float | None = None
    youngs_modulus: float | None = None
    poissons_ratio: float | None = None
    stretch_stiffness: float | None = None
    shear_stiffness: float | None = None
    bend_stiffness: float | None = None


@dataclass(frozen=True)
class NewtonSurfaceMaterialValues:
    """Converted values accepted by ``ModelBuilder.add_cloth_mesh``."""

    thickness: float
    density: float | None
    particle_radius: float
    tri_ke: float | None
    tri_ka: float | None
    edge_ke: float | None
    ignored_shear_stiffness: float | None
    used_default_thickness: bool


def _optional_nonnegative(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}.")
    return value


def convert_surface_material(inputs: SurfaceMaterialInputs) -> NewtonSurfaceMaterialValues:
    """Convert OmniPhysics thin-shell values to Newton isotropic cloth values.

    The conversion intentionally mirrors the coarse mapping used by Newton's
    USD surface-deformable importer:

    * volume density becomes area density through ``rho_area = rho * h``;
    * a positive stretch override becomes ``tri_ke = stretch * h``;
    * a positive bend override becomes ``edge_ke = bend * h**3``;
    * particle collision radius starts at half the physical thickness.

    OmniPhysics defines zero stretch/bend values as "derive from Young's
    modulus and thickness". For that fallback, a plane-stress isotropic
    membrane is used. Newton's triangle parameters then correspond to the 2-D
    Lame parameters integrated over shell thickness.

    Independent shear stiffness cannot be represented by Newton's isotropic
    ``add_cloth_mesh`` interface. It is returned for an explicit warning by
    the caller, rather than being silently folded into another coefficient.
    """

    thickness = _optional_nonnegative("surface thickness", inputs.thickness)
    used_default_thickness = thickness is None or thickness == 0.0
    if used_default_thickness:
        thickness = DEFAULT_SURFACE_THICKNESS

    density = _optional_nonnegative("surface density", inputs.density)
    if density == 0.0:
        density = None

    youngs_modulus = _optional_nonnegative("Young's modulus", inputs.youngs_modulus)
    if youngs_modulus == 0.0:
        youngs_modulus = None

    poissons_ratio = inputs.poissons_ratio
    if poissons_ratio is not None:
        poissons_ratio = float(poissons_ratio)
        if not math.isfinite(poissons_ratio) or not 0.0 <= poissons_ratio < 0.5:
            raise ValueError(
                "Poisson's ratio for surface conversion must be finite and in [0, 0.5), "
                f"got {poissons_ratio!r}."
            )

    stretch_stiffness = _optional_nonnegative("surface stretch stiffness", inputs.stretch_stiffness)
    shear_stiffness = _optional_nonnegative("surface shear stiffness", inputs.shear_stiffness)
    bend_stiffness = _optional_nonnegative("surface bend stiffness", inputs.bend_stiffness)

    active_stretch_override = stretch_stiffness is not None and stretch_stiffness > 0.0
    active_bend_override = bend_stiffness is not None and bend_stiffness > 0.0

    tri_ke = None
    tri_ka = None
    if active_stretch_override:
        # Newton treats tri_ke as the in-plane shear/Lame-mu term integrated
        # over the triangle's surface area.
        tri_ke = stretch_stiffness * thickness
        if poissons_ratio is None:
            # This is the behavior of Newton 1.4's proposal importer when the
            # source material carries no independent Poisson term.
            tri_ka = 0.0
        else:
            tri_ka = 2.0 * tri_ke * poissons_ratio / (1.0 - poissons_ratio)
    elif youngs_modulus is not None:
        nu = 0.25 if poissons_ratio is None else poissons_ratio
        tri_ke = youngs_modulus * thickness / (2.0 * (1.0 + nu))
        tri_ka = youngs_modulus * nu * thickness / (1.0 - nu * nu)

    edge_ke = None
    if active_bend_override:
        edge_ke = bend_stiffness * thickness**3
    elif youngs_modulus is not None:
        nu = 0.25 if poissons_ratio is None else poissons_ratio
        edge_ke = youngs_modulus * thickness**3 / (12.0 * (1.0 - nu * nu))

    ignored_shear_stiffness = (
        shear_stiffness if shear_stiffness is not None and shear_stiffness > 0.0 else None
    )
    return NewtonSurfaceMaterialValues(
        thickness=thickness,
        density=None if density is None else density * thickness,
        particle_radius=0.5 * thickness,
        tri_ke=tri_ke,
        tri_ka=tri_ka,
        edge_ke=edge_ke,
        ignored_shear_stiffness=ignored_shear_stiffness,
        used_default_thickness=used_default_thickness,
    )


__all__ = [
    "DEFAULT_SURFACE_THICKNESS",
    "NewtonSurfaceMaterialValues",
    "SurfaceMaterialInputs",
    "convert_surface_material",
]
