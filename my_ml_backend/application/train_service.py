"""Training application service.

Provides training job orchestration on top of backend adapters and job storage.
"""

from typing import Any, Dict

from backend_core.domain.routing import build_route_spec
from backend_core.domain.training import TrainRequest


class TrainService:
    """Coordinates training requests and job lifecycle transitions."""

    def __init__(self, backend_registry, family_exists, job_store, logger):
        self.backend_registry = backend_registry
        self.family_exists = family_exists
        self.job_store = job_store
        self.logger = logger

    def fit(self, event: str, data: Dict[str, Any], **kwargs):
        """Create and execute a training job for the resolved model route.

        Returns a serialized job snapshot containing status, timestamps,
        route metadata, and optional training result.
        """
        data = data or {}
        params = dict(kwargs)

        model_name = params.get('model_name') or data.get('model_name')
        model_task = params.get('model_task') or data.get('model_task') or 'detect'
        model_family = params.get('model_family') or data.get('model_family')
        route_spec = build_route_spec(model_name, model_task, model_family)

        request = TrainRequest(
            event=str(event or 'TRAINING_START'),
            data=data,
            params=params,
            model_name=route_spec.model_name,
            model_task=route_spec.model_task,
            model_family=route_spec.model_family,
            backend=route_spec.backend,
        )
        job = self.job_store.create(request)

        if not self.family_exists(route_spec.model_family):
            self.logger.error("Unsupported model family for training: %s", route_spec.model_family)
            job = self.job_store.update(
                job.job_id,
                status='failed',
                message=f"Unsupported model family: {route_spec.model_family}",
            )
            return self._serialize_job(job)

        adapter = self.backend_registry.get_adapter(route_spec.backend)
        if adapter is None:
            self.logger.error("Backend unavailable for training: %s", route_spec.backend)
            job = self.job_store.update(
                job.job_id,
                status='failed',
                message=f"Backend unavailable: {route_spec.backend}",
            )
            return self._serialize_job(job)

        if not getattr(adapter, 'supports_training', False):
            message = f"Training is not supported for backend: {route_spec.backend}"
            self.logger.warning(message)
            job = self.job_store.update(job.job_id, status='failed', message=message)
            return self._serialize_job(job)

        self.job_store.update(job.job_id, status='running', message='Training started')
        try:
            train_result = adapter.train(request=request, job_id=job.job_id)
            job = self.job_store.update(
                job.job_id,
                status='succeeded',
                message='Training completed',
                result=train_result if isinstance(train_result, dict) else {'result': train_result},
            )
        except Exception as exc:
            self.logger.exception("Training failed for job_id=%s", job.job_id)
            job = self.job_store.update(
                job.job_id,
                status='failed',
                message=str(exc),
            )
        return self._serialize_job(job)

    def _serialize_job(self, job):
        """Convert internal ``TrainJob`` entity to API-facing dictionary."""
        return {
            'job_id': job.job_id,
            'status': job.status,
            'message': job.message,
            'result': job.result,
            'model_family': job.request.model_family,
            'model_task': job.request.model_task,
            'model_name': job.request.model_name,
            'backend': job.request.backend,
            'created_at': job.request.created_at,
            'updated_at': job.updated_at,
        }
