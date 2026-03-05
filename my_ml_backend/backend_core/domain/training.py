from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TrainRequest:
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
    job_id: str
    status: str
    request: TrainRequest
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    updated_at: str = field(default_factory=_utcnow_iso)
