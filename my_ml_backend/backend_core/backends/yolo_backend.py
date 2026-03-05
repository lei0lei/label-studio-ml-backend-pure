from typing import Dict, Optional
from .base import BackendAdapter


try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class YoloBackendAdapter(BackendAdapter):
    backend_name = 'yolo'
    supports_training = False

    @property
    def model_cls(self):
        return YOLO

    def run(self, selected_model, local_path: str, model_task: str, imgsz: int, context: Optional[Dict], task: Dict):
        infer_kwargs = {'imgsz': imgsz}
        if model_task == 'segment':
            infer_kwargs['retina_masks'] = True
        return selected_model(local_path, **infer_kwargs)
