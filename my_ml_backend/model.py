
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
import os
import logging
from backend_core.domain.routing import (
    family_exists,
    is_task_supported,
)
from backend_core.backends.registry import BackendRegistry
from backend_core.model_loader import ModelLoader
from backend_core.inference_runner import InferenceRunner
from presentation.result_builders import LabelStudioResultBuilder
from presentation.tag_mapper import TagMapper
from infrastructure.path_resolver import PathResolver
from infrastructure.job_store import InMemoryJobStore
from application.predict_service import PredictService
from application.train_service import TrainService


logger = logging.getLogger(__name__)
class Model(LabelStudioMLBase):
    """Unified Label Studio ML Backend with model-family + task routing"""

    def setup(self):
        default_model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
        self.default_model_path = os.getenv('YOLOV8_MODEL_PATH', default_model_path)
        sam2_default = os.getenv('SAM2_MODEL_PATH', os.path.join(os.path.dirname(__file__), 'sam2.1_hiera_tiny.pt'))
        self.default_sam2_model_path = sam2_default if os.path.exists(sam2_default) else self.default_model_path
        self.model_dir = os.getenv('YOLOV8_MODEL_DIR', os.path.dirname(self.default_model_path))
        logger.info(
            "Model router initialized: model_dir=%s default_model=%s default_sam2=%s",
            self.model_dir,
            self.default_model_path,
            self.default_sam2_model_path,
        )
        self.backend_registry = BackendRegistry()
        self.model_loader = ModelLoader(
            model_dir=self.model_dir,
            default_model_path=self.default_model_path,
            default_sam2_model_path=self.default_sam2_model_path,
        )
        self.inference_runner = InferenceRunner(
            backend_registry=self.backend_registry,
            logger=logger,
        )
        self.result_builder = LabelStudioResultBuilder(logger=logger)
        self.tag_mapper = TagMapper(logger=logger)
        self.path_resolver = PathResolver(logger=logger)
        self.job_store = InMemoryJobStore()
        self.predict_service = PredictService(
            model_loader=self.model_loader,
            backend_registry=self.backend_registry,
            family_exists=family_exists,
            is_task_supported=is_task_supported,
            resolve_tag_mapping=self._resolve_tag_mapping,
            resolve_local_path=self._resolve_local_path,
            run_inference=self.inference_runner.run,
            postprocess=self._postprocess,
            logger=logger,
        )
        self.train_service = TrainService(
            backend_registry=self.backend_registry,
            family_exists=family_exists,
            job_store=self.job_store,
            logger=logger,
        )
        self.set("model_version", "model-router-1.0")
        self.from_name = "label"
        self.to_name = "image"
        self.labels_in_config = []
        self.result_type = "rectanglelabels"
        self.result_label_key = "rectanglelabels"

    def _resolve_tag_mapping(self, model_task: str):
        mapping = self.tag_mapper.resolve(
            model_task=model_task,
            get_first_tag_occurence=self.get_first_tag_occurence,
            parsed_label_config=self.parsed_label_config,
            fallback_from_name=self.from_name,
            fallback_to_name=self.to_name,
        )
        self.from_name = mapping.from_name
        self.to_name = mapping.to_name
        self.labels_in_config = mapping.labels_in_config
        self.result_type = mapping.result_type
        self.result_label_key = mapping.result_label_key
    def _postprocess(self, inference_result, model_task: str):
        return self.result_builder.build(
            inference_result=inference_result,
            model_task=model_task,
            from_name=self.from_name,
            to_name=self.to_name,
            labels_in_config=self.labels_in_config,
            result_type=self.result_type,
        )

    def _resolve_local_path(self, image_url: str, task_id=None):
        return self.path_resolver.resolve(
            image_url=image_url,
            task_id=task_id,
            get_local_path=self.get_local_path,
        )

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        return self.predict_service.predict(tasks=tasks, context=context, **kwargs)

    def fit(self, event, data, **kwargs):
        return self.train_service.fit(event=event, data=data, **kwargs)


