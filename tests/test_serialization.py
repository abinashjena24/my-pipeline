
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.serialization import load_pipeline, save_pipeline


@pytest.fixture
def fitted_pipeline() -> Pipeline:
    X = pd.DataFrame({"feature1": [1.0, 2.0, 3.0, 4.0]})
    y = np.array([0, 0, 1, 1])

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression()),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def test_save_and_load_pipeline(
    tmp_path: Path,
    fitted_pipeline: Pipeline,
) -> None:
    artifact_path = tmp_path / "artifacts" / "model.joblib"

    save_pipeline(fitted_pipeline, artifact_path)
    loaded_pipeline = load_pipeline(artifact_path)

    X = pd.DataFrame({"feature1": [1.5, 3.5]})

    np.testing.assert_array_equal(
        fitted_pipeline.predict(X),
        loaded_pipeline.predict(X),
    )
    assert artifact_path.is_file()


def test_load_pipeline_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.joblib"

    with pytest.raises(FileNotFoundError):
        load_pipeline(missing_path)

