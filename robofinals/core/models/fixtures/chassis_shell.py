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

from .fixture import Fixture
from .fixture_types import FixtureType


class ChassisShell(Fixture):
    fixture_types = [FixtureType.CHASSISSHELL]

    def __init__(self, name, prim, num_envs, **kwargs):
        super().__init__(name, prim, num_envs, **kwargs)

    def get_reset_region_names(self):
        """Return parsed chassis shell interior region names."""
        int_regions = tuple(
            sorted(
                name
                for name in self._regions
                if name.startswith("int") and name != "int"
            )
        )
        return int_regions or super().get_reset_region_names()
