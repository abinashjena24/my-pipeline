from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class RunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)

    random_seed: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    hyperparameters: dict[str, object]
    metrics: dict[str, float]
    artifact_path: Path
