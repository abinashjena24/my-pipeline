from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inference import (
    run_inference,
    validate_batch,
    validate_cols,
    validate_paths,
)


class DummyModel:
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.ones(len(df), dtype=int)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.tile([0.4, 0.6], (len(df), 1))


def test_validate_paths_accepts_existing_file(tmp_path: Path) -> None:
    input_file = tmp_path / "input.csv"
    input_file.write_text("feature1,feature2\n1,10\n", encoding="utf-8")

    validate_paths(input_file)


def test_validate_paths_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_paths(tmp_path / "missing.csv")


def test_validate_cols_rejects_missing_column() -> None:
    df = pd.DataFrame({"feature1": [1]})

    with pytest.raises(ValueError, match="Missing"):
        validate_cols(df, ["feature1", "feature2"])


def test_validate_cols_rejects_extra_column() -> None:
    df = pd.DataFrame({"feature1": [1], "feature2": [10], "unexpected": [99]})

    with pytest.raises(ValueError, match="Unexpected"):
        validate_cols(df, ["feature1", "feature2"])


def test_run_inference_adds_predictions_and_probabilities() -> None:
    df = pd.DataFrame({"feature1": [21, 22], "feature2": [210, 220]})

    results = run_inference(df, DummyModel())  # type: ignore[arg-type]

    assert results["prediction"].tolist() == [1, 1]
    assert results["probability_class_1"].tolist() == [0.6, 0.6]


def test_validate_batch_reports_extra_field() -> None:
    df = pd.DataFrame({"feature1": [1], "feature2": [10], "unexpected": [99]})

    valid, invalid = validate_batch(df)

    assert valid.empty
    assert len(invalid) == 1
    assert "error" in invalid.columns


def test_validate_batch_accepts_missing_cells() -> None:
    df = pd.DataFrame(
        {
            "feature1": [1.0, float("nan")],
            "feature2": [10.0, 20.0],
        }
    )

    valid, invalid = validate_batch(df)

    assert len(valid) == 2
    assert invalid.empty
    assert pd.isna(valid.loc[1, "feature1"])


def test_validate_batch_reports_invalid_row() -> None:
    df = pd.DataFrame(
        {
            "feature1": [1.0, "bad"],
            "feature2": [10.0, 20.0],
        }
    )

    valid, invalid = validate_batch(df)

    assert len(valid) == 1
    assert len(invalid) == 1
    assert invalid.loc[0, "row_index"] == 1
    assert "error" in invalid.columns


def test_validate_batch_preserves_original_row_indices() -> None:
    df = pd.DataFrame(
        {
            "feature1": [1, "bad", 3],
            "feature2": [10, 20, 30],
        },
        index=[10, 20, 30],
    )

    valid, invalid = validate_batch(df)

    assert valid["row_index"].tolist() == [10, 30]
    assert invalid["row_index"].tolist() == [20]


def test_run_inference_preserves_row_count_and_columns() -> None:
    df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [10, 20, 30]})

    results = run_inference(df, DummyModel())  # type: ignore[arg-type]

    assert len(results) == len(df)
    assert results.columns.tolist() == [
        "feature1",
        "feature2",
        "prediction",
        "probability_class_1",
    ]


def test_validate_batch_rejects_unexpected_category() -> None:
    df = pd.DataFrame({"feature1": ["unknown"], "feature2": [10]})

    valid, invalid = validate_batch(df)

    assert valid.empty
    assert len(invalid) == 1
    assert "error" in invalid.columns


def test_run_inference_preserves_input_dataframe() -> None:
    df = pd.DataFrame({"feature1": [1, 2], "feature2": [10, 20]})
    original = df.copy(deep=True)

    run_inference(df, DummyModel())  # type: ignore[arg-type]

    pd.testing.assert_frame_equal(df, original)
