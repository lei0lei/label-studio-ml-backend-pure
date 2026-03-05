"""Training domain entities and timestamps."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _utcnow_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TrainRequest:
    """Training command payload captured at request time."""

    event: str
    data: Dict[str, Any]
    params: Dict[str, Any]
    model_name: str
    model_task: str
    model_family: str
    backend: str
    created_at: str = field(default_factory=_utcnow_iso)


@dataclass
class TrainJob:
    """Long-running training job state tracked by storage layer."""

    job_id: str
    status: str
    request: TrainRequest
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    updated_at: str = field(default_factory=_utcnow_iso)
