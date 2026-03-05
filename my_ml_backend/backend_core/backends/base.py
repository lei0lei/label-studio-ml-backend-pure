"""Backend adapter base contract."""

from typing import Dict, Optional


class BackendAdapter:
    """Abstract interface implemented by concrete backend adapters."""

    backend_name = 'base'
    supports_training = False

    @property
    def model_cls(self):
        """Return callable/class used to load model artifacts."""
        raise NotImplementedError

    def run(self, selected_model, local_path: str, model_task: str, imgsz: int, context: Optional[Dict], task: Dict):
        """Execute inference and return backend-native result list."""
        raise NotImplementedError

    def train(self, request, job_id: str):
        """Execute backend-specific training routine."""
        raise NotImplementedError
