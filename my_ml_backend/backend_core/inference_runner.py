"""Backend inference execution adapter.

Maps route information to a backend adapter and delegates runtime inference.
"""

from backend_core.domain.routing import build_route_spec


class InferenceRunner:
    """Thin coordinator for invoking backend adapter inference."""

    def __init__(self, backend_registry, logger):
        self.backend_registry = backend_registry
        self.logger = logger

    def run(self, selected_model, local_path: str, model_task: str, model_family: str, imgsz: int, context, task):
        """Execute model inference through the resolved backend adapter."""
        route_spec = build_route_spec('best', model_task, model_family)
        adapter = self.backend_registry.get_adapter(route_spec.backend)
        if adapter is None:
            self.logger.error("No backend adapter found for backend=%s", route_spec.backend)
            return []
        return adapter.run(
            selected_model=selected_model,
            local_path=local_path,
            model_task=model_task,
            imgsz=imgsz,
            context=context,
            task=task,
        )
