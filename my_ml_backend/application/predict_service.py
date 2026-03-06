"""Prediction application service.

This module orchestrates model route resolution, model loading, path resolution,
backend inference, and Label Studio response assembly.
"""

from typing import Dict, List, Optional

from label_studio_ml.response import ModelResponse
from backend_core.domain.routing import build_route_spec


class PredictService:
    """Application-layer facade for batch prediction requests."""

    def __init__(
        self,
        model_loader,
        backend_registry,
        family_exists,
        is_task_supported,
        resolve_tag_mapping,
        resolve_local_path,
        run_inference,
        postprocess,
        logger,
    ):
        self.model_loader = model_loader
        self.backend_registry = backend_registry
        self.family_exists = family_exists
        self.is_task_supported = is_task_supported
        self.resolve_tag_mapping = resolve_tag_mapping
        self.resolve_local_path = resolve_local_path
        self.run_inference = run_inference
        self.postprocess = postprocess
        self.logger = logger

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Run end-to-end prediction for Label Studio tasks.

        The method resolves routing parameters from ``kwargs`` and ``context``,
        validates compatibility, executes per-task inference, and returns a
        ``ModelResponse`` payload that Label Studio can consume directly.
        """
        model_name = kwargs.get('model_name')
        if not model_name and isinstance(context, dict):
            model_name = context.get('model_name')
        model_task = kwargs.get('model_task')
        if not model_task and isinstance(context, dict):
            model_task = context.get('model_task')
        model_family = kwargs.get('model_family')
        if not model_family and isinstance(context, dict):
            model_family = context.get('model_family')

        route_spec = build_route_spec(model_name, model_task, model_family)
        model_name = route_spec.model_name
        model_task = route_spec.model_task
        model_family = route_spec.model_family
        imgsz = route_spec.imgsz

        if not self.family_exists(model_family):
            self.logger.error("Unsupported model family: %s", model_family)
            return ModelResponse(predictions=[])
        if not self.is_task_supported(model_family, model_task):
            self.logger.error("Unsupported task=%s for family=%s", model_task, model_family)
            return ModelResponse(predictions=[])

        self.resolve_tag_mapping(model_task=model_task)

        self.logger.info(
            "Predict request received: tasks=%d model_task=%s model_family=%s model_name=%s",
            len(tasks),
            model_task,
            model_family,
            model_name,
        )

        model_version = f"{model_task}/{model_family}/{model_name}"
        selected_model = self.model_loader.get_or_load_model(route_spec, self.backend_registry)
        self.logger.info(
            "Using model_task=%s model_family=%s model_name=%s",
            model_task,
            model_family,
            model_name
        )

        predictions = []

        def _has_sam2_features(ctx):
            if not isinstance(ctx, dict):
                return False
            direct_features = ctx.get('sam2_features')
            if isinstance(direct_features, dict):
                return True

            result_items = ctx.get('result')
            if not isinstance(result_items, list):
                return False
            for item in result_items:
                if not isinstance(item, dict):
                    continue
                item_meta = item.get('meta')
                if isinstance(item_meta, dict) and isinstance(item_meta.get('sam2_features'), dict):
                    return True
            return False

        context_has_sam2_features = _has_sam2_features(context)
        for task in tasks:
            task_id = task.get('id')
            image_url = task['data'].get('image') or task['data'].get('image_url')
            self.logger.info("Processing task_id=%s image_url=%s", task_id, image_url)
            if not image_url:
                self.logger.warning("task_id=%s has no image url", task_id)
                predictions.append({"result": [], "score": 0.0, "model_version": model_version})
                continue

            if selected_model is None:
                self.logger.warning("task_id=%s skipped because model is not available", task_id)
                predictions.append({"result": [], "score": 0.0, "model_version": model_version})
                continue

            local_path = ''
            if not (model_family == 'sam2' and context_has_sam2_features):
                try:
                    local_path = self.resolve_local_path(image_url, task_id=task_id)
                except Exception as exc:
                    self.logger.error("task_id=%s failed to resolve local image path: %s", task_id, exc)
                    predictions.append({"result": [], "score": 0.0, "model_version": model_version})
                    continue
            else:
                self.logger.info('task_id=%s uses SAM2 feature payload, skip local image resolution', task_id)

            results = self.run_inference(
                selected_model=selected_model,
                local_path=local_path,
                model_task=model_task,
                model_family=model_family,
                imgsz=imgsz,
                context=context,
                task=task,
            )
            inference_result = results[0] if results else None
            if inference_result is None:
                predictions.append({"result": [], "score": 0.0, "model_version": model_version})
                continue

            result = self.postprocess(inference_result, model_task=model_task)

            self.logger.info(
                "Inference done for task_id=%s: boxes=%d, labels=%s",
                task_id,
                len(result),
                [
                    (r.get("value", {}).get("rectanglelabels") or r.get("value", {}).get("polygonlabels") or ["?"])[0]
                    for r in result
                ][:10]
            )
            predictions.append({
                "result": result,
                "score": float(max((float(r.get("score", 0.0)) for r in result), default=0.0)),
                "model_version": model_version
            })

        self.logger.info("Predict response ready: predictions=%d", len(predictions))
        return ModelResponse(predictions=predictions)
