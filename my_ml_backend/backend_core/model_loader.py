import logging
import os
import re
from typing import Optional


logger = logging.getLogger(__name__)


class ModelLoader:
    _loaded_models = {}

    def __init__(self, model_dir: str, default_model_path: str, default_sam2_model_path: str):
        self.model_dir = model_dir
        self.default_model_path = default_model_path
        self.default_sam2_model_path = default_sam2_model_path

    def _sanitize_model_name(self, model_name: str):
        if not model_name:
            return 'best'
        safe_name = re.sub(r'[^0-9a-zA-Z_\-.]', '', str(model_name))
        return safe_name or 'best'

    def _sanitize_model_segment(self, segment: str):
        if not segment:
            return None
        safe_segment = re.sub(r'[^0-9a-zA-Z_\-]', '', str(segment))
        return safe_segment or None

    def _resolve_model_path(self, model_name: str, model_task: Optional[str] = None, model_family: Optional[str] = None, backend: Optional[str] = None):
        safe_name = self._sanitize_model_name(model_name)
        safe_task = self._sanitize_model_segment(model_task)
        safe_family = self._sanitize_model_segment(model_family)

        if safe_name in ('default', 'best'):
            cache_key = f"{safe_task or 'default'}::{safe_family or 'default'}::{safe_name}"
            if backend == 'sam2':
                return self.default_sam2_model_path, cache_key
            return self.default_model_path, cache_key

        candidate_dirs = []
        if safe_task and safe_family:
            candidate_dirs.append(os.path.join(self.model_dir, safe_task, safe_family))
        candidate_dirs.append(self.model_dir)

        model_filename = f"{safe_name}.pt"
        for candidate_dir in candidate_dirs:
            model_path = os.path.join(candidate_dir, model_filename)
            if os.path.exists(model_path):
                cache_key = f"{safe_task or 'default'}::{safe_family or 'default'}::{safe_name}"
                return model_path, cache_key

        logger.warning(
            "Model file not found for model_task=%s model_family=%s model_name=%s, fallback to default model",
            safe_task,
            safe_family,
            safe_name,
        )
        fallback_key = f"{safe_task or 'default'}::{safe_family or 'default'}::best"
        return self.default_model_path, fallback_key

    def get_or_load_model(self, route_spec, backend_registry):
        adapter = backend_registry.get_adapter(route_spec.backend)
        if adapter is None:
            logger.error("Backend unavailable for family=%s", route_spec.model_family)
            return None

        model_cls = adapter.model_cls
        if model_cls is None:
            logger.error("Model class unavailable for backend=%s", route_spec.backend)
            return None

        model_path, cache_key = self._resolve_model_path(
            model_name=route_spec.model_name,
            model_task=route_spec.model_task,
            model_family=route_spec.model_family,
            backend=route_spec.backend,
        )
        full_cache_key = f"{route_spec.backend}::{cache_key}"
        if full_cache_key in self.__class__._loaded_models:
            return self.__class__._loaded_models[full_cache_key]

        logger.info("Lazy loading model backend=%s key=%s path=%s", route_spec.backend, full_cache_key, model_path)
        loaded_model = model_cls(model_path)
        self.__class__._loaded_models[full_cache_key] = loaded_model
        return loaded_model
