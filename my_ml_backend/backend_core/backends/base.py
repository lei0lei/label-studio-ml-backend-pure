from typing import Dict, Optional


class BackendAdapter:
    backend_name = 'base'
    supports_training = False

    @property
    def model_cls(self):
        raise NotImplementedError

    def run(self, selected_model, local_path: str, model_task: str, imgsz: int, context: Optional[Dict], task: Dict):
        raise NotImplementedError

    def train(self, request, job_id: str):
        raise NotImplementedError
