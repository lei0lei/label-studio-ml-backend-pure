"""In-memory training job repository.

Provides lightweight lifecycle storage for training jobs during backend runtime.
"""

import uuid
from datetime import datetime, timezone

from backend_core.domain.training import TrainJob


def _utcnow_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


class InMemoryJobStore:
    """Store and update training jobs in process memory."""

    def __init__(self):
        self._jobs = {}

    def create(self, request) -> TrainJob:
        """Create a queued training job for the supplied request entity."""
        job_id = str(uuid.uuid4())
        job = TrainJob(job_id=job_id, status='queued', request=request)
        self._jobs[job_id] = job
        return job

    def update(self, job_id: str, status: str, message=None, result=None) -> TrainJob:
        """Update an existing job state and return the mutated entity."""
        job = self._jobs[job_id]
        job.status = status
        job.message = message
        job.result = result
        job.updated_at = _utcnow_iso()
        return job

    def get(self, job_id: str):
        """Fetch job by identifier, or ``None`` when absent."""
        return self._jobs.get(job_id)
