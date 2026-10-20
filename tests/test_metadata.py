
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.metadata import RunMetadata


def test_run_metadata_valid_data() -> None:
    metadata = RunMetadata(
        run_id="test-run-001",
        model_name="random_forest",
        random_seed=42,
        hyperparameters={"n_estimators": 100},
        metrics={"accuracy": 0.85, "roc_auc": 0.90},
        artifact_path=Path("artifacts/test-run-001/model.joblib"),
    )

    assert metadata.run_id == "test-run-001"
    assert metadata.model_name == "random_forest"
    assert metadata.random_seed == 42
    assert isinstance(metadata.created_at, datetime)
    assert metadata.created_at.tzinfo == timezone.utc


def test_run_metadata_rejects_missing_fields() -> None:
    with pytest.raises(ValidationError):
        RunMetadata( # type: ignore[call-arg]
            run_id="test-run-002",
            model_name="random_forest",
            random_seed=42,
            hyperparameters={},
            metrics={},
        )


def test_run_metadata_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RunMetadata( # type: ignore[call-arg]
            run_id="test-run-003",
            model_name="random_forest",
            random_seed=42,
            hyperparameters={},
            metrics={},
            artifact_path=Path("model.joblib"),
            unexpected_field="not allowed",
        )
