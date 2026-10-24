from typing import Any

import pandas as pd
import pytest

from src.benchmark import METRIC_NAMES, MODEL_NAMES, run_benchmark


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "feature1": list(range(1, 41)),
            "feature2": list(range(10, 410, 10)),
        }
    )
    y = pd.Series([0, 1] * 20)
    return X, y


def test_benchmark_returns_all_models_and_metrics(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = sample_data

    params: dict[str, dict[str, Any]] = {
        "random_forest": {"n_estimators": 10, "random_state": 42},
        "xgboost": {
            "n_estimators": 10,
            "random_state": 42,
            "eval_metric": "logloss",
        },
        "lightgbm": {
            "n_estimators": 10,
            "random_state": 42,
            "verbosity": -1,
        },
    }

    results = run_benchmark(X, y, params, cv_folds=5, random_seed=42)

    assert set(results["model"]) == set(MODEL_NAMES)
    assert len(results) == 3

    for metric in METRIC_NAMES:
        assert f"{metric}_mean" in results.columns
        assert f"{metric}_std" in results.columns

    assert results["roc_auc_mean"].between(0, 1).all()
    assert results["roc_auc_std"].ge(0).all()


def test_benchmark_rejects_too_few_samples() -> None:
    X = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y = pd.Series([0, 0, 0, 1])

    with pytest.raises(ValueError, match="at least cv_folds"):
        run_benchmark(X, y, {}, cv_folds=5)
