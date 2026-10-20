
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.artifacts import save_artifacts
from src.metadata import RunMetadata


def test_save_artifacts(tmp_path: Path) -> None:
    X = pd.DataFrame({"feature1": [1.0, 2.0, 3.0, 4.0]})
    y = [0, 0, 1, 1]

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression()),
        ]
    )
    pipeline.fit(X, y)

    metadata = RunMetadata(
        run_id="test-run-001",
        model_name="logistic_regression",
        random_seed=42,
        hyperparameters={},
        metrics={"accuracy": 1.0},
        artifact_path=Path("model.joblib"),
    )

    run_dir = save_artifacts(
        pipeline=pipeline,
        metadata=metadata,
        output_dir=tmp_path / "artifacts",
    )

    model_path = run_dir / "model.joblib"
    metadata_path = run_dir / "metadata.json"

    assert model_path.is_file()
    assert metadata_path.is_file()

    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert saved_metadata["run_id"] == "test-run-001"
    assert saved_metadata["model_name"] == "logistic_regression"
    assert saved_metadata["artifact_path"] == str(model_path)
