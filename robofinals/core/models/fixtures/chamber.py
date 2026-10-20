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

from isaaclab.envs import ManagerBasedRLEnv

from .fixture import Fixture
from .fixture_types import FixtureType


class Chamber(Fixture):
    """Fixture model for the chamber asset under ``/World/Chamber``.

    The authored asset has one controllable lid joint, ``Lid001_joint``, with
    a range of -90 to 0 degrees.  The base Fixture implementation normalizes
    that range so 0 means closed and 1 means open.
    """

    fixture_types = [FixtureType.CHAMBER]
    LID_JOINT_NAME = "Lid001_joint"

    def setup_env(self, env: ManagerBasedRLEnv):
        super().setup_env(env)
        self._env = env

    @property
    def lid_joint_names(self):
        """Joint names controlling the chamber lid, resolved from the USD."""
        authored = [name for name in self._joint_infos if "lid" in name.lower()]
        return authored or [self.LID_JOINT_NAME]

    @property
    def door_joint_names(self):
        """Compatibility alias used by generic Fixture door helpers."""
        return self.lid_joint_names

    def _require_lid_articulation(self, env):
        articulation = env.scene.articulations.get(self.name)
        if articulation is None:
            raise KeyError(
                f"Chamber articulation {self.name!r} is not registered; "
                f"available={tuple(env.scene.articulations.keys())}"
            )
        available = tuple(articulation.data.joint_names)
        missing = [name for name in self.lid_joint_names if name not in available]
        if missing:
            raise KeyError(
                f"Chamber {self.name!r} is missing lid joints {missing}; "
                f"available={available}"
            )
        return articulation

    def is_lid_open(self, env, th=0.9):
        """Return a boolean tensor indicating whether the lid is open."""
        self._require_lid_articulation(env)
        return super().is_open(env, joint_names=self.lid_joint_names, th=th)

    def is_lid_closed(self, env, th=0.005):
        """Return a boolean tensor indicating whether the lid is closed."""
        self._require_lid_articulation(env)
        return super().is_closed(env, joint_names=self.lid_joint_names, th=th)

    def is_open(self, env, th=0.9):
        """Compatibility alias for :meth:`is_lid_open`."""
        return self.is_lid_open(env, th=th)

    def is_closed(self, env, th=0.005):
        """Compatibility alias for :meth:`is_lid_closed`."""
        return self.is_lid_closed(env, th=th)

    def check_asset_contact(self, env, asset_name, threshold=1.0e-8):
        """Return whether an asset is contacting the chamber.

        The pair query is delegated to the active backend.  PhysX resolves a
        filtered rigid-contact view; Newton uses its declared contact-pair
        sensors, so Newton tasks should declare ``(asset_name, self.name)`` in
        ``pairwise_contacts`` when using this interface.
        """
        del threshold  # Contact providers own backend-specific thresholds.
        from robofinals.utils import object_utils as object_utils

        return object_utils.check_contact(env, asset_name, self)

    def is_asset_contact(self, env, asset_name, threshold=1.0e-8):
        """Readable alias for :meth:`check_asset_contact`."""
        return self.check_asset_contact(env, asset_name, threshold=threshold)

    def check_contact(self, env, asset_name, threshold=1.0e-8):
        """Short alias for :meth:`check_asset_contact`."""
        return self.check_asset_contact(env, asset_name, threshold=threshold)

    def update_state(self, env: ManagerBasedRLEnv):
        """Chamber has no powered state; lid state is read directly from its joint."""
        return None

    def get_state(self):
        """Return the fixture state maintained outside the articulation buffers."""
        return {}
