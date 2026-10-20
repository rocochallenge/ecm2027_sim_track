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

"""Make ``import pinocchio`` resolve to the robotics library (cmeel ``pin`` wheel).

A different PyPI project also publishes a top-level ``pinocchio`` package (nose
plugins) without ``SE3`` / ``robot_wrapper``. The ``pin`` wheel installs the
real bindings under ``site-packages/cmeel.prefix/...`` which may appear on
``sys.path`` after plain ``site-packages``, so the stub wins unless cmeel is
prepended and a bad cached module is dropped.
"""

from __future__ import annotations

import os
import sys


def _cmeel_pinocchio_site() -> str:
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    return os.path.join(
        sys.prefix,
        "lib",
        f"python{ver}",
        "site-packages",
        "cmeel.prefix",
        "lib",
        f"python{ver}",
        "site-packages",
    )


def ensure_robotics_pinocchio_loaded() -> None:
    """Prefer cmeel pinocchio over a namesake ``site-packages/pinocchio`` stub."""
    vp = _cmeel_pinocchio_site()
    if not os.path.isdir(vp):
        return
    try:
        sys.path.remove(vp)
    except ValueError:
        pass
    sys.path.insert(0, vp)

    mod = sys.modules.get("pinocchio")
    if mod is not None and not hasattr(mod, "SE3"):
        del sys.modules["pinocchio"]
