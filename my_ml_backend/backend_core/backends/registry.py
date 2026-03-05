"""Backend adapter registry.

Provides lookup of runtime adapters by backend key.
"""

from .yolo_backend import YoloBackendAdapter
from .sam2_backend import Sam2BackendAdapter


class BackendRegistry:
    """Container for backend adapter instances."""

    def __init__(self):
        self._adapters = {
            'yolo': YoloBackendAdapter(),
            'sam2': Sam2BackendAdapter(),
        }

    def get_adapter(self, backend: str):
        """Return adapter instance by backend key, or ``None`` if missing."""
        return self._adapters.get(backend)
