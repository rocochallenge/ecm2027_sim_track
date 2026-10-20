from __future__ import annotations

from typing import Any
from robofinals.core.backends.contract import ContactSensorRef


class ContactQuery:
    """Logical contact port; backend views stay behind the provider."""

    def __init__(self, provider: Any) -> None:
        self.provider = provider

    def has_contact(self, source: ContactSensorRef | str, target: ContactSensorRef | str, *, threshold: float = 1e-8) -> Any:
        return self.provider.has_contact(source, target, threshold=threshold)

    def read_contact(self, sensor: ContactSensorRef | str, *, strict: bool = False) -> Any:
        return self.provider.read_contact(sensor, strict=strict)
