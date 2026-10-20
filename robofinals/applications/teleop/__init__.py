"""Vertically sliceable teleoperation application."""

import os

# This package is initialized before teleop.entrypoint. Pin native thread
# pools before importing modules that load Torch, NumPy, or Pinocchio.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from .application import TeleopApplication
from .commands import CommandMailbox
from .devices import DeviceContext, DeviceRegistry, TeleopDeviceBinding, build_default_device_registry
from .driver import LegacyDeviceActionSource, TeleopDriver
from .garment_runtime import GarmentManager
from .spec import TeleopSpec

__all__ = [
    "CommandMailbox",
    "DeviceContext",
    "DeviceRegistry",
    "LegacyDeviceActionSource",
    "GarmentManager",
    "TeleopApplication",
    "TeleopDeviceBinding",
    "TeleopDriver",
    "TeleopSpec",
    "build_default_device_registry",
]
