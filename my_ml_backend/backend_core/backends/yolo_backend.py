"""YOLO backend adapter implementation."""

from typing import Dict, Optional
from .base import BackendAdapter


try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class YoloBackendAdapter(BackendAdapter):
    """Adapter that delegates inference to Ultralytics YOLO models."""

    backend_name = 'yolo'
    supports_training = False

    @property
    def model_cls(self):
        """Return YOLO model class used for model loading."""
        return YOLO

    def run(self, selected_model, local_path: str, model_task: str, imgsz: int, context: Optional[Dict], task: Dict):
        """Execute YOLO inference with task-specific runtime options."""
        infer_kwargs = {'imgsz': imgsz}
        if model_task == 'segment':
            infer_kwargs['retina_masks'] = True
        return selected_model(local_path, **infer_kwargs)
