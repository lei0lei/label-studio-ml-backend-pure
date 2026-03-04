from .yolo_backend import YoloBackendAdapter
from .sam2_backend import Sam2BackendAdapter


class BackendRegistry:
    def __init__(self):
        self._adapters = {
            'yolo': YoloBackendAdapter(),
            'sam2': Sam2BackendAdapter(),
        }

    def get_adapter(self, backend: str):
        return self._adapters.get(backend)
