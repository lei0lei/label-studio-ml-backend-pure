import uuid
from datetime import datetime, timezone

from backend_core.domain.training import TrainJob


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class InMemoryJobStore:
    def __init__(self):
        self._jobs = {}

    def create(self, request) -> TrainJob:
        job_id = str(uuid.uuid4())
        job = TrainJob(job_id=job_id, status='queued', request=request)
        self._jobs[job_id] = job
        return job

    def update(self, job_id: str, status: str, message=None, result=None) -> TrainJob:
        job = self._jobs[job_id]
        job.status = status
        job.message = message
        job.result = result
        job.updated_at = _utcnow_iso()
        return job

    def get(self, job_id: str):
        return self._jobs.get(job_id)
