"""Public compatibility name for the Isaac startup bootstrap.

The implementation lives in :mod:`bootstrap`; this module deliberately
contains no simulation loop. New code should compose :class:`TeleopApplication`.
"""

from .bootstrap import TeleopSession

__all__ = ["TeleopSession"]
