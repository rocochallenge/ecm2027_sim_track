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

from robofinals.utils.usd_utils import OpenUsd as usd

from .fixture import Fixture
from .fixture_types import FixtureType


class Shelf(Fixture):
    fixture_types = [FixtureType.SHELF]

    def __init__(self, name, prim, num_envs, **kwargs):
        super().__init__(name, prim, num_envs, **kwargs)


class GoodsShelf(Fixture):
    fixture_types = [FixtureType.GOODSSHELF]

    def __init__(self, name, prim, num_envs, **kwargs):
        super().__init__(name, prim, num_envs, **kwargs)

    def get_reset_region_names(self):
        return ("int2", "int1")


class Storagebin(Fixture):
    fixture_types = [FixtureType.STORAGEBIN]

    def __init__(self, name, prim, num_envs, **kwargs):
        super().__init__(name, prim, num_envs, **kwargs)

    @property
    def contact_prim(self):
        """Rigid child prim used by contact sensors."""
        pending = [self.prim]
        while pending:
            prim = pending.pop(0)
            if usd.is_rigidbody(prim):
                return prim
            pending.extend(prim.GetAllChildren())
        return self.prim


class Workbench(Fixture):
    fixture_types = [FixtureType.WORKBENCH]

    def __init__(self, name, prim, num_envs, **kwargs):
        super().__init__(name, prim, num_envs, **kwargs)

    def get_reset_region_names(self):
        return ("int", "int0")


class Cabinetry(Fixture):
    fixture_types = [FixtureType.CABINETRY]

    def __init__(self, name, prim, num_envs, **kwargs):
        super().__init__(name, prim, num_envs, **kwargs)
        self.int_names = ("int", "int13")

    def get_reset_region_names(self):
        return self.int_names

    def set_int_reg_name(self, name_tuple):
        self.int_names = name_tuple


class Tray(Fixture):
    fixture_types = [FixtureType.TRAY]

    def __init__(self, name, prim, num_envs, **kwargs):
        super().__init__(name, prim, num_envs, **kwargs)

    def get_reset_region_names(self):
        return ("int", "int0")
